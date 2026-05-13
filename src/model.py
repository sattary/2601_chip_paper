import torch
import torch.nn as nn
import torch.nn.functional as F

class HINN_MultiTask(nn.Module):
    """
    Hardware-Informed Neural Network (HINN).
    
    Rationale:
    A shared backbone learns the joint representation of the HLS configuration.
    Divergent heads predict Area and Latency separately, forcing the network
    to map the Pareto conflict internally. This architecture allows us to compute
    the physics-informed constraint losses efficiently.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int] = [256, 128, 64]):
        super().__init__()
        
        # Shared Trunk
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim
            
        self.shared_trunk = nn.Sequential(*layers)
        
        # Area Head (Predicts hls_lut, hls_ff)
        self.area_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )
        
        # Latency Head (Predicts average_latency, best_latency)
        # Note: Depending on target selection, output dim can be 1 or 2
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        shared_features = self.shared_trunk(x)
        area_preds = self.area_head(shared_features)
        latency_preds = self.latency_head(shared_features)
        
        # Concatenate back to match the target vector (Area, Latency)
        return torch.cat([area_preds, latency_preds], dim=-1)


def hinn_loss(preds_real, preds_perturbed, y_real, lambda_area=0.1, lambda_lat=0.1):
    """
    Computes the loss: L_Data + L_Constraints.
    
    Args:
        preds_real: Predictions on the actual configuration.
        preds_perturbed: Predictions on the synthetically perturbed configuration (higher parallelism).
        y_real: Ground truth labels.
        lambda_area: Weight for the area monotonicity penalty.
        lambda_lat: Weight for the latency monotonicity penalty.
    """
    # 1. Standard Data Loss (MSE)
    l_data = F.mse_loss(preds_real, y_real)
    
    # Extract specific predictions (Assuming Target Columns are: LUT, FF, Avg_Lat, Best_Lat)
    # Area = LUT (index 0) + FF (index 1) [or handle them independently]
    area_real = preds_real[:, 0] + preds_real[:, 1]
    area_pert = preds_perturbed[:, 0] + preds_perturbed[:, 1]
    
    lat_real = preds_real[:, 2] 
    lat_pert = preds_perturbed[:, 2]
    
    # 2. Physics Penalty: Area Monotonicity
    # If parallelism increases, Area MUST increase. (Area_pert >= Area_real)
    # So if Area_real > Area_pert, apply penalty.
    l_area_mono = torch.mean(F.relu(area_real - area_pert))
    
    # 3. Physics Penalty: Latency Monotonicity
    # If parallelism increases, Latency MUST decrease. (Lat_pert <= Lat_real)
    # So if Lat_pert > Lat_real, apply penalty.
    l_lat_mono = torch.mean(F.relu(lat_pert - lat_real))
    
    # Total Loss
    loss = l_data + (lambda_area * l_area_mono) + (lambda_lat * l_lat_mono)
    
    return loss, l_data, l_area_mono, l_lat_mono
