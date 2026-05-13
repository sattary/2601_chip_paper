import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class HINNDataset(Dataset):
    """
    Hardware-Informed Neural Network (HINN) Dataset.
    
    Rationale:
    Standard PyTorch Datasets return (X, y) pairs. To enforce the discrete 
    monotonicity constraints required for the "Physics" penalty without needing 
    ground-truth labels for every possible configuration, this loader dynamically 
    generates an (X, X_perturbed, y) triplet. X_perturbed represents the exact 
    same hardware design, but with a synthetically increased parallelism factor.
    The HINN loss function will penalize the network if X_perturbed does not 
    predict a higher Area and lower Latency than X.
    """
    def __init__(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, parallelism_prefix: str = 'param_3_'):
        """
        Args:
            features_df: One-hot encoded feature matrix (X).
            targets_df: Target matrix (y).
            parallelism_prefix: The prefix of the one-hot columns corresponding 
                                to the parallelism parameter we want to perturb.
        """
        self.X = torch.tensor(features_df.values, dtype=torch.float32)
        self.y = torch.tensor(targets_df.values, dtype=torch.float32)
        self.columns = features_df.columns.tolist()
        
        # Identify the indices of the one-hot columns that represent the parallelism factor
        # e.g., 'param_3_4', 'param_3_8', 'param_3_16'
        self.parallel_cols = [
            (idx, int(col.replace(parallelism_prefix, ''))) 
            for idx, col in enumerate(self.columns) 
            if col.startswith(parallelism_prefix) and col.replace(parallelism_prefix, '').isdigit()
        ]
        
        # Sort by the factor value (e.g., 4, 8, 16)
        self.parallel_cols.sort(key=lambda x: x[1])

    def __len__(self):
        return len(self.X)

    def _perturb_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synthetically increases the parallelism factor of the feature vector.
        If it's already at max parallelism, we leave it (loss difference will be 0).
        """
        x_pert = x.clone()
        
        # Find which parallel factor is currently active (value == 1)
        current_idx = -1
        for i, (col_idx, factor) in enumerate(self.parallel_cols):
            if x_pert[col_idx] > 0.5:  # It's one-hot
                current_idx = i
                break
                
        # If we found an active factor and it's not the maximum one
        if current_idx != -1 and current_idx < len(self.parallel_cols) - 1:
            # Turn off current factor
            x_pert[self.parallel_cols[current_idx][0]] = 0.0
            # Turn on the next higher factor
            x_pert[self.parallel_cols[current_idx + 1][0]] = 1.0
            
        return x_pert

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        x_perturbed = self._perturb_features(x)
        
        return x, x_perturbed, y
