"""
HINN Training Pipeline.

Entry point for train_hinn(). All orchestration logic lives here; model
architecture is in model.py and dataset batching in dataset.py.
"""
import random
import os
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from dataset import HINNDataset
from model import HINN_MultiTask, hinn_loss


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int) -> None:
    """
    Fix all stochastic sources used in this pipeline.

    Rationale: A single seed produces a single-point estimate. Callers should
    run this function with multiple seeds (e.g., 42, 0, 7, 123, 999) and report
    mean ± std to satisfy Expert Systems statistical rigor requirements.
    torch.backends.cudnn.deterministic slows CUDA training slightly but ensures
    bit-exact reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# TARGET TRANSFORM UTILITIES
# =============================================================================

def log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to all columns.

    Rationale: Both area (LUT skew=7.9, FF skew=11.8) and latency (skew=7.9,
    CV=4.97) are severely right-skewed. Raw StandardScaler maps the dominant
    low-resource region (~75% of data) into a narrow negative band while the
    long tail occupies the full positive range. This collapses MSE gradient
    magnitude where it matters most. log1p compresses the tail and spreads the
    body, making the loss geometry well-conditioned. log1p is preferred over log
    because it is defined at 0, avoiding an offset hack.
    """
    return np.log1p(df)


def inverse_log1p_transform(arr: np.ndarray) -> np.ndarray:
    """Inverse of log1p: expm1. Recovers physical units at inference time."""
    return np.expm1(arr)


# =============================================================================
# PHYSICS METRIC
# =============================================================================

def calculate_cvr(preds_real: torch.Tensor, preds_pert: torch.Tensor) -> float:
    """
    Constraint Violation Ratio (CVR).

    Percentage of (x, x_pert) pairs where the predicted direction violates the
    parallelism-scaling prior. CVR is computed in transformed (log) space because
    the model operates there. Monotonicity in log-space implies monotonicity in
    physical space (log1p is monotone).

    Denominator is total_samples * 2 because each sample contributes two
    independent constraint checks (area and latency).
    """
    area_real = preds_real[:, 0] + preds_real[:, 1]
    area_pert = preds_pert[:, 0] + preds_pert[:, 1]
    lat_real = preds_real[:, 2]
    lat_pert = preds_pert[:, 2]

    area_violations = (area_real > area_pert).float().sum()
    lat_violations = (lat_pert > lat_real).float().sum()

    total = preds_real.shape[0] * 2
    return ((area_violations + lat_violations) / total).item() * 100.0


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================

def train_hinn(
    epochs: int = 200,
    batch_size: int = 1024,
    seed: int = 42,
    loko_kernel: str | None = None,
) -> None:
    """
    Train the HINN surrogate and persist weights + scalers.

    For multi-seed statistical reporting (required by Expert Systems):
    run this function with seed in [42, 0, 7, 123, 999] via:
        uv run python cli.py train train --seed <N> --epochs 300
    """
    set_global_seed(seed)
    print(f"Seed: {seed}  |  Epochs: {epochs}  |  Batch: {batch_size}")

    print("Loading data...")
    features_df = pd.read_parquet("data/processed/features.parquet")
    targets_df = pd.read_parquet("data/processed/targets.parquet")

    # Handle Data Splitting
    if loko_kernel:
        print(f"Applying LOKO validation: leaving out kernel '{loko_kernel}'")
        if 'kernel_name' not in targets_df.columns:
            raise ValueError("kernel_name not found in targets.parquet. Re-run data_prep.py.")
            
        test_mask = targets_df['kernel_name'] == loko_kernel
        X_test = features_df[test_mask]
        y_test = targets_df[test_mask]
        
        X_train_val = features_df[~test_mask]
        y_train_val = targets_df[~test_mask]
        
        # Further split train_val into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=seed
        )
    else:
        # 80/10/10 split — test set is held out and never touched during training
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_df, targets_df, test_size=0.1, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=1 / 9, random_state=seed
        )
        
    # Strip metadata columns from targets (keep only numerical columns for training)
    # Target columns must be exactly the 4 we expect
    target_cols = ['hls_lut', 'hls_ff', 'average_latency', 'best_latency']
    y_train = y_train[target_cols]
    y_val = y_val[target_cols]
    y_test = y_test[target_cols]

    print(f"Splits — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Scalers: area and latency both receive log1p before StandardScaler.
    # This is applied to train only; val/test use the fitted transformers.
    # ------------------------------------------------------------------
    scaler_X = StandardScaler()
    scaler_y_area = StandardScaler()
    scaler_y_lat = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler_X.fit_transform(X_train), columns=X_train.columns
    )

    # Area: log1p before fit (LUT skew=7.9, FF skew=11.8)
    y_train_area_log = log1p_transform(y_train.iloc[:, :2])
    y_train_area_scaled = scaler_y_area.fit_transform(y_train_area_log)

    # Latency: log1p before fit (avg_lat CV=4.97)
    y_train_lat_log = log1p_transform(y_train.iloc[:, 2:])
    y_train_lat_scaled = scaler_y_lat.fit_transform(y_train_lat_log)

    y_train_scaled = pd.DataFrame(
        np.hstack([y_train_area_scaled, y_train_lat_scaled]),
        columns=y_train.columns,
    )

    # Val: mirror transforms (no refit)
    X_val_scaled = pd.DataFrame(
        scaler_X.transform(X_val), columns=X_val.columns
    )
    y_val_area_scaled = scaler_y_area.transform(log1p_transform(y_val.iloc[:, :2]))
    y_val_lat_scaled = scaler_y_lat.transform(log1p_transform(y_val.iloc[:, 2:]))
    y_val_scaled = pd.DataFrame(
        np.hstack([y_val_area_scaled, y_val_lat_scaled]),
        columns=y_val.columns,
    )

    # Persist scalers + transform config sentinel
    # transform_config drives inference-time inversion in moo.py and visualize/
    out_dir = Path("results/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, out_dir / "scaler_X.pkl")
    joblib.dump(scaler_y_area, out_dir / "scaler_y_area.pkl")
    joblib.dump(scaler_y_lat, out_dir / "scaler_y_lat.pkl")
    joblib.dump(
        {"area_log1p_applied": True, "lat_log1p_applied": True},
        out_dir / "transform_config.pkl",
    )
    print("Scalers saved.")

    # Datasets and loaders
    train_dataset = HINNDataset(X_train_scaled, y_train_scaled, parallelism_prefix="param_3_")
    val_dataset = HINNDataset(X_val_scaled, y_val_scaled, parallelism_prefix="param_3_")
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HINN_MultiTask(
        input_dim=X_train.shape[1],
        feature_names=list(X_train.columns),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing: smooth LR decay from 1e-3 to 1e-5 over T_max epochs.
    # Rationale: flat LR (prior code) wastes budget in the late training phase
    # where fine-grained gradient steps matter for constraint learning.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")

    print(
        f"\n{'Epoch':<6} | {'LR':<8} | {'Lambda':<6} | "
        f"{'Train MSE':<10} | {'Val MSE':<10} | {'Val R2':<8} | {'CVR (%)':<8}"
    )
    print("-" * 75)

    metrics_file = out_dir / "metrics.csv"
    with open(metrics_file, "w") as f:
        f.write("epoch,lr,lambda_weight,train_mse,val_mse,val_r2,val_cvr\n")

    # Rationale for phased lambda schedule:
    # 1. 0-30: Warm-up (lambda=0) to establish basic mapping.
    # 2. 30-100: Soft-ramp to 0.3 (soft Bayesian prior).
    # 3. 100-300: Plateau at 0.3 to maximize R2 performance.
    # 4. 300-500: Hard-ramp to 0.8 to force strict physical proof in final epochs.
    PHASE1_MAX = 0.3
    PHASE2_MAX = 0.8
    RAMP_START = 30
    RAMP1_END = 100
    RAMP2_START = 300
    RAMP2_END = 500

    for epoch in range(1, epochs + 1):
        if epoch <= RAMP_START:
            lambda_weight = 0.0
        elif epoch <= RAMP1_END:
            lambda_weight = PHASE1_MAX * (epoch - RAMP_START) / (RAMP1_END - RAMP_START)
        elif epoch <= RAMP2_START:
            lambda_weight = PHASE1_MAX
        elif epoch <= RAMP2_END:
            # Linear ramp from 0.3 to 0.8
            lambda_weight = PHASE1_MAX + (PHASE2_MAX - PHASE1_MAX) * (epoch - RAMP2_START) / (RAMP2_END - RAMP2_START)
        else:
            lambda_weight = PHASE2_MAX

        # Train
        model.train()
        train_mse_total = 0.0
        for x, x_pert, y in train_loader:
            x, x_pert, y = x.to(device), x_pert.to(device), y.to(device)
            optimizer.zero_grad()
            loss, mse_loss, _, _ = hinn_loss(
                model(x), model(x_pert), y,
                lambda_area=lambda_weight,
                lambda_lat=lambda_weight,
            )
            loss.backward()
            optimizer.step()
            train_mse_total += mse_loss.item() * x.size(0)

        scheduler.step()
        train_mse = train_mse_total / len(train_dataset)
        current_lr = scheduler.get_last_lr()[0]

        # Validate
        model.eval()
        val_mse_total = 0.0
        all_preds: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        cvr_accum = 0.0

        with torch.no_grad():
            for x, x_pert, y in val_loader:
                x, x_pert, y = x.to(device), x_pert.to(device), y.to(device)
                preds = model(x)
                preds_pert = model(x_pert)
                _, mse_loss, _, _ = hinn_loss(preds, preds_pert, y, lambda_area=0, lambda_lat=0)
                val_mse_total += mse_loss.item() * x.size(0)
                cvr_accum += calculate_cvr(preds, preds_pert) * x.size(0)
                all_preds.append(preds.cpu().numpy())
                all_y.append(y.cpu().numpy())

        val_mse = val_mse_total / len(val_dataset)
        val_cvr = cvr_accum / len(val_dataset)
        val_r2 = r2_score(np.vstack(all_y), np.vstack(all_preds))

        with open(metrics_file, "a") as f:
            f.write(f"{epoch},{current_lr:.6f},{lambda_weight},{train_mse},{val_mse},{val_r2},{val_cvr}\n")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:<6} | {current_lr:<8.5f} | {lambda_weight:<6.2f} | "
                f"{train_mse:<10.4f} | {val_mse:<10.4f} | {val_r2:<8.4f} | {val_cvr:<8.2f}"
            )

        # Save best model — only after physics constraints are meaningfully active
        if val_mse < best_val_loss and lambda_weight > 0.1:
            best_val_loss = val_mse
            torch.save(model.state_dict(), out_dir / "hinn_best.pt")

    print(f"\nTraining complete. Best val MSE (lambda>0.1): {best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Test set evaluation — the number that goes into the paper table.
    # Never touched during any training or early stopping decision.
    # ------------------------------------------------------------------
    best_model_path = out_dir / "hinn_best.pt"
    if best_model_path.exists():
        print("\nEvaluating on held-out test set...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        X_test_scaled = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)
        y_test_area_scaled = scaler_y_area.transform(log1p_transform(y_test.iloc[:, :2]))
        y_test_lat_scaled = scaler_y_lat.transform(log1p_transform(y_test.iloc[:, 2:]))
        y_test_scaled_arr = np.hstack([y_test_area_scaled, y_test_lat_scaled])

        X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32).to(device)

        with torch.no_grad():
            test_preds = model(X_test_t).cpu().numpy()

        test_r2 = r2_score(y_test_scaled_arr, test_preds)
        # Per-target R² for the results table
        per_target_r2 = {
            col: r2_score(y_test_scaled_arr[:, i], test_preds[:, i])
            for i, col in enumerate(["hls_lut", "hls_ff", "average_latency", "best_latency"])
        }
        print(f"Test R² (all targets): {test_r2:.4f}")
        for k, v in per_target_r2.items():
            print(f"  {k}: {v:.4f}")

        # Persist test results for comparison table
        test_results_path = out_dir / "test_results.csv"
        rows = [{"seed": seed, "model": "HINN", "split": "test", "target": k, "r2": v}
                for k, v in per_target_r2.items()]
        rows.append({"seed": seed, "model": "HINN", "split": "test", "target": "all", "r2": test_r2})
        pd.DataFrame(rows).to_csv(test_results_path, index=False)
        print(f"Test results saved to {test_results_path}")
    else:
        print("Warning: hinn_best.pt not found (lambda > 0.1 was never met). No test eval.")


if __name__ == "__main__":
    train_hinn()
