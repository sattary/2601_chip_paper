import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.bn2(self.fc2(x))
        x = x + res
        return F.gelu(x)


class HINN_MultiTask(nn.Module):
    """
    Hardware-Informed Neural Network (HINN).

    Rationale (Option B - ResNet):
    Entity Embeddings (Option A) gave the model too much capacity and destroyed the
    ordinal relationship between parallelism factors, causing massive overfitting
    when the physics constraint turned on.
    Instead, we use a ResNet trunk. Skip connections allow the physics constraint 
    gradients to flow directly to the earlier representations without destroying 
    the data-fitting features, preventing R2 collapse.
    """

    def __init__(
        self,
        input_dim: int,
        feature_names: list[str] | None = None,  # Ignored, kept for API compat
        embed_dim: int = 16,                     # Ignored
        hidden_dims: list[int] | None = None,    # Ignored, we use fixed ResNet dims
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        hidden_dim = 256
        num_blocks = 3

        # Initial projection to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ResNet Trunk
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(hidden_dim, dropout))
        self.shared_trunk = nn.Sequential(*blocks)

        # Area Head: predicts hls_lut (idx 0) and hls_ff (idx 1)
        self.area_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),
        )

        # Latency Head: predicts average_latency (idx 2) and best_latency (idx 3)
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
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

