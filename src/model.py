import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HINN_MultiTask(nn.Module):
    """
    Hardware-Informed Neural Network (HINN).

    Rationale:
    A shared backbone learns the joint representation of the HLS configuration.
    Divergent heads predict Area and Latency separately, forcing the network to
    map the Pareto conflict internally. Dropout(0.1) prevents co-adaptation on
    rare one-hot categories in the sparse 262-dim input. The 512-first-layer
    avoids aggressive early compression of nearly-orthogonal OHE directions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            # 512 entry gives the trunk room to disambiguate the ~262 OHE directions
            # before the bottleneck compression sequence.
            hidden_dims = [512, 256, 128, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        self.shared_trunk = nn.Sequential(*layers)

        # Area Head: predicts hls_lut (idx 0) and hls_ff (idx 1)
        self.area_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

        # Latency Head: predicts average_latency (idx 2) and best_latency (idx 3)
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_trunk(x)
        return torch.cat([self.area_head(shared), self.latency_head(shared)], dim=-1)


def hinn_loss(
    preds_real: torch.Tensor,
    preds_perturbed: torch.Tensor,
    y_real: torch.Tensor,
    lambda_area: float = 0.1,
    lambda_lat: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes L_total = L_data + lambda_area * L_area_mono + lambda_lat * L_lat_mono.

    The monotonicity penalties are hinge losses (ReLU-gated violations).
    They function as soft regularizers encoding the parallelism-scaling prior:
    more unrolling => more area, less latency. The prior holds empirically for
    ~52% of intra-kernel pairs in db4hls; lambda annealing prevents it from
    dominating before the data loss has converged.

    Args:
        preds_real:      Predictions for the base configuration. Shape (B, 4).
        preds_perturbed: Predictions for the higher-parallelism perturbation. Shape (B, 4).
        y_real:          Ground truth targets (in transformed space). Shape (B, 4).
        lambda_area:     Penalty weight for area monotonicity violation.
        lambda_lat:      Penalty weight for latency monotonicity violation.

    Returns:
        Tuple of (total_loss, l_data, l_area_mono, l_lat_mono).
    """
    l_data: torch.Tensor = F.mse_loss(preds_real, y_real)

    # Combined area proxy: LUT + FF (indices 0, 1)
    area_real = preds_real[:, 0] + preds_real[:, 1]
    area_pert = preds_perturbed[:, 0] + preds_perturbed[:, 1]

    # Average latency (index 2; in log-space after transform)
    lat_real = preds_real[:, 2]
    lat_pert = preds_perturbed[:, 2]

    # Penalty: area_pert should be >= area_real (more parallelism -> more LUTs)
    l_area_mono: torch.Tensor = torch.mean(F.relu(area_real - area_pert))

    # Penalty: lat_pert should be <= lat_real (more parallelism -> fewer cycles)
    l_lat_mono: torch.Tensor = torch.mean(F.relu(lat_pert - lat_real))

    loss = l_data + (lambda_area * l_area_mono) + (lambda_lat * l_lat_mono)
    return loss, l_data, l_area_mono, l_lat_mono

