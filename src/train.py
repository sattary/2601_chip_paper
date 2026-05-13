import os
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from dataset import HINNDataset
from model import HINN_MultiTask, hinn_loss

def calculate_cvr(preds_real, preds_pert):
    """
    Constraint Violation Ratio (CVR).
    Percentage of predictions that mathematically violate Amdahl's Law or area scaling bounds.
    """
    area_real = preds_real[:, 0] + preds_real[:, 1]
    area_pert = preds_pert[:, 0] + preds_pert[:, 1]
    lat_real = preds_real[:, 2]
    lat_pert = preds_pert[:, 2]
    
    # Violation: Area should increase with more parallelism (Area_pert >= Area_real)
    area_violations = (area_real > area_pert).float().sum()
    
    # Violation: Latency should decrease with more parallelism (Lat_pert <= Lat_real)
    lat_violations = (lat_pert > lat_real).float().sum()
    
    total_samples = preds_real.shape[0] * 2  # Two constraints per sample
    total_violations = area_violations + lat_violations
    
    return (total_violations / total_samples).item() * 100.0

def train_hinn(epochs: int = 200, batch_size: int = 32):
    print("Loading data...")
    features_df = pd.read_parquet('data/processed/features.parquet')
    targets_df = pd.read_parquet('data/processed/targets.parquet')
    
    # Split: 80% Train, 10% Val, 10% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(features_df, targets_df, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=42) # 1/9 of 0.9 is 0.1
    
    print(f"Data Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scalers
    scaler_X = StandardScaler()
    scaler_y_area = StandardScaler()
    scaler_y_lat = StandardScaler()
    
    # Fit & Transform Train
    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), columns=X_train.columns)
    
    y_train_area_scaled = scaler_y_area.fit_transform(y_train.iloc[:, :2])
    y_train_lat_scaled = scaler_y_lat.fit_transform(y_train.iloc[:, 2:])
    y_train_scaled = pd.DataFrame(np.hstack([y_train_area_scaled, y_train_lat_scaled]), columns=y_train.columns)
    
    # Transform Val
    X_val_scaled = pd.DataFrame(scaler_X.transform(X_val), columns=X_val.columns)
    y_val_area_scaled = scaler_y_area.transform(y_val.iloc[:, :2])
    y_val_lat_scaled = scaler_y_lat.transform(y_val.iloc[:, 2:])
    y_val_scaled = pd.DataFrame(np.hstack([y_val_area_scaled, y_val_lat_scaled]), columns=y_val.columns)
    
    # Save Scalers
    out_dir = Path("results/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, out_dir / 'scaler_X.pkl')
    joblib.dump(scaler_y_area, out_dir / 'scaler_y_area.pkl')
    joblib.dump(scaler_y_lat, out_dir / 'scaler_y_lat.pkl')
    print("Scalers saved.")

    # Datasets & Loaders
    # Assuming 'param_3_' is the primary unroll/parallelism factor based on our analysis
    train_dataset = HINNDataset(X_train_scaled, y_train_scaled, parallelism_prefix='param_3_')
    val_dataset = HINNDataset(X_val_scaled, y_val_scaled, parallelism_prefix='param_3_')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    model = HINN_MultiTask(input_dim=X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    print("\nStarting Training...")
    print(f"{'Epoch':<6} | {'Lambda':<6} | {'Train MSE':<10} | {'Val MSE':<10} | {'Val R2':<8} | {'CVR (%)':<8}")
    print("-" * 65)
    
    for epoch in range(1, epochs + 1):
        # Lambda Annealing Schedule
        if epoch <= 50:
            lambda_weight = 0.0
        elif epoch <= 150:
            lambda_weight = (epoch - 50) / 100.0
        else:
            lambda_weight = 1.0
            
        # Train Loop
        model.train()
        train_mse_total = 0
        for x, x_pert, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            preds_pert = model(x_pert)
            
            loss, mse_loss, l_area, l_lat = hinn_loss(
                preds, preds_pert, y, 
                lambda_area=lambda_weight, 
                lambda_lat=lambda_weight
            )
            
            loss.backward()
            optimizer.step()
            train_mse_total += mse_loss.item() * x.size(0)
            
        train_mse = train_mse_total / len(train_dataset)
        
        # Validation Loop
        model.eval()
        val_mse_total = 0
        all_preds = []
        all_y = []
        cvr_total = 0
        
        with torch.no_grad():
            for x, x_pert, y in val_loader:
                preds = model(x)
                preds_pert = model(x_pert)
                
                _, mse_loss, _, _ = hinn_loss(preds, preds_pert, y, lambda_area=0, lambda_lat=0)
                val_mse_total += mse_loss.item() * x.size(0)
                
                cvr_total += calculate_cvr(preds, preds_pert) * x.size(0)
                
                all_preds.append(preds.numpy())
                all_y.append(y.numpy())
                
        val_mse = val_mse_total / len(val_dataset)
        val_cvr = cvr_total / len(val_dataset)
        
        # Calculate R2
        all_preds = np.vstack(all_preds)
        all_y = np.vstack(all_y)
        val_r2 = r2_score(all_y, all_preds)
        
        # Log to metrics.csv
        metrics_file = out_dir / 'metrics.csv'
        if epoch == 1:
            with open(metrics_file, 'w') as f:
                f.write('epoch,lambda_weight,train_mse,val_mse,val_r2,val_cvr\n')
        with open(metrics_file, 'a') as f:
            f.write(f'{epoch},{lambda_weight},{train_mse},{val_mse},{val_r2},{val_cvr}\n')
        
        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:<6} | {lambda_weight:<6.2f} | {train_mse:<10.4f} | {val_mse:<10.4f} | {val_r2:<8.4f} | {val_cvr:<8.2f}")
            
        # Save Best Model
        if val_mse < best_val_loss and lambda_weight > 0.5: # only save once physics constraints are active
            best_val_loss = val_mse
            torch.save(model.state_dict(), out_dir / 'hinn_best.pt')

    print(f"\nTraining Complete. Best Val MSE (under physics constraints): {best_val_loss:.4f}")
    print("Model saved to results/models/hinn_best.pt")

if __name__ == "__main__":
    train_hinn()
