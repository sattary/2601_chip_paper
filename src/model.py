import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HINN_MultiTask(nn.Module):
    """
    Hardware-Informed Neural Network (HINN).

    Rationale:
    Option A (Entity Embeddings): Instead of passing 262 flat, sparse OHE bits into
    the trunk, we group the one-hot columns by their parent categorical parameter
    (e.g., param_0, param_1) and pass each group through an nn.Linear projection.
    Mathematically, nn.Linear applied to an OHE vector is identical to an
    nn.Embedding lookup, but this allows us to use the existing data pipeline.
    This creates a dense, semantically meaningful representation of HLS kernels
    (e.g. FFT vs GEMM) before the compression bottleneck, boosting Area R2.

    A shared backbone learns the joint representation of the HLS configuration.
    Divergent heads predict Area and Latency separately.
    """

    def __init__(
        self,
        input_dim: int,
        feature_names: list[str] | None = None,
        embed_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.use_embeddings = feature_names is not None
        
        if self.use_embeddings:
            from collections import defaultdict
            self.param_groups = defaultdict(list)
            for i, col in enumerate(feature_names):
                parts = col.split('_')
                if len(parts) >= 2 and parts[0] == 'param':
                    group_name = f'param_{parts[1]}'
                    self.param_groups[group_name].append(i)
                else:
                    self.param_groups['other'].append(i)
            
            self.embeddings = nn.ModuleDict()
            trunk_input_dim = 0
            for group_name, indices in self.param_groups.items():
                group_dim = len(indices)
                # Provide a reasonable embedding capacity for categorical variables
                out_dim = min(embed_dim, group_dim) if group_dim > 1 else group_dim
                self.embeddings[group_name] = nn.Linear(group_dim, out_dim)
                trunk_input_dim += out_dim
                self.register_buffer(f"idx_{group_name}", torch.tensor(indices, dtype=torch.long))
        else:
            trunk_input_dim = input_dim

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        layers: list[nn.Module] = []
        prev_dim = trunk_input_dim
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
        if self.use_embeddings:
            embedded_chunks = []
            for group_name, linear_layer in self.embeddings.items():
                idx = getattr(self, f"idx_{group_name}")
                chunk = x[:, idx]
                embedded = linear_layer(chunk)
                embedded_chunks.append(embedded)
            shared_input = torch.cat(embedded_chunks, dim=-1)
            shared_input = F.gelu(shared_input)
        else:
            shared_input = x
            
        shared = self.shared_trunk(shared_input)
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

