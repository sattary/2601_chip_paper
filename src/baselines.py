"""
Baseline Models for HINN Comparison.

Provides two baselines trained on the identical data pipeline as HINN:

  A. XGBoost (MultiOutputRegressor) — the canonical HLS DSE surrogate baseline
     per Bai et al. 2021 (db4hls paper). Represents the state-of-practice
     non-DL approach.

  B. Vanilla MLP — same HINN_MultiTask architecture but with lambda=0 (no
     physics penalty) and a standard TensorDataset (no perturbation batching).
     Ablates the physics constraint mechanism: if HINN does not outperform
     this, the physics loss provides no benefit.

Both models use the same log1p + StandardScaler transform pipeline as HINN.
Results are written to results/models/baseline_results.csv in the same
per-target R² format as test_results.csv so a single comparison table can
be assembled.
"""

import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import sys

# Allow importing model from parent src directory
sys.path.insert(0, str(Path(__file__).parent))
from model import HINN_MultiTask
from train import log1p_transform, set_global_seed


TARGET_COLS = ["hls_lut", "hls_ff", "average_latency", "best_latency"]


# =============================================================================
# SHARED DATA LOADING
# =============================================================================

def _load_splits(seed: int) -> tuple[
    pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame,
]:
    """Load parquet and produce the same 80/10/10 splits as train_hinn."""
    features_df = pd.read_parquet("data/processed/features.parquet")
    targets_df = pd.read_parquet("data/processed/targets.parquet")

    X_tv, X_test, y_tv, y_test = train_test_split(
        features_df, targets_df, test_size=0.1, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=1 / 9, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_scalers(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> tuple[StandardScaler, StandardScaler, StandardScaler]:
    """Fit the same three scalers (with log1p) as HINN training."""
    scaler_X = StandardScaler()
    scaler_y_area = StandardScaler()
    scaler_y_lat = StandardScaler()

    scaler_X.fit_transform(X_train)
    scaler_y_area.fit_transform(log1p_transform(y_train.iloc[:, :2]))
    scaler_y_lat.fit_transform(log1p_transform(y_train.iloc[:, 2:]))

    return scaler_X, scaler_y_area, scaler_y_lat


def _transform(
    X: pd.DataFrame,
    y: pd.DataFrame,
    scaler_X: StandardScaler,
    scaler_y_area: StandardScaler,
    scaler_y_lat: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = scaler_X.transform(X)
    y_area = scaler_y_area.transform(log1p_transform(y.iloc[:, :2]))
    y_lat = scaler_y_lat.transform(log1p_transform(y.iloc[:, 2:]))
    return X_scaled, np.hstack([y_area, y_lat])


def _per_target_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {col: r2_score(y_true[:, i], y_pred[:, i]) for i, col in enumerate(TARGET_COLS)}


# =============================================================================
# BASELINE A: XGBOOST
# =============================================================================

def train_xgboost(
    seed: int = 42,
    n_estimators: int = 500,
    max_depth: int = 7,
    learning_rate: float = 0.05,
) -> dict[str, float]:
    """
    Train a MultiOutputRegressor(XGBRegressor) baseline.

    Rationale: XGBoost is the de-facto standard in HLS DSE surrogate literature
    (Bai et al. 2021, AutoDSE, FlexPar). It handles mixed categorical+numeric
    features well and requires no normalisation, making it a strong non-DL
    baseline. We wrap in MultiOutputRegressor to produce all 4 targets.

    Returns per-target test R² dict.
    """
    from xgboost import XGBRegressor

    print(f"\n[XGBoost] seed={seed}, n_estimators={n_estimators}, max_depth={max_depth}")
    set_global_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test = _load_splits(seed)
    scaler_X, scaler_y_area, scaler_y_lat = _build_scalers(X_train, y_train)

    # XGBoost handles raw features well; we still apply log1p to targets for
    # consistency — it dramatically reduces RMSE on the skewed targets.
    X_tr, y_tr = _transform(X_train, y_train, scaler_X, scaler_y_area, scaler_y_lat)
    _, y_val_t = _transform(X_val, y_val, scaler_X, scaler_y_area, scaler_y_lat)
    X_te, y_te = _transform(X_test, y_test, scaler_X, scaler_y_area, scaler_y_lat)

    base = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=20,
    )
    model = MultiOutputRegressor(base, n_jobs=1)

    # XGBoost within MultiOutputRegressor does not support eval_set natively,
    # so we train each target model separately for early stopping, then repack.
    print("  Training 4 XGBoost models (one per target) with early stopping...")
    models = []
    val_r2_list = []
    for i, col in enumerate(TARGET_COLS):
        m = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=20,
        )
        m.fit(X_tr, y_tr[:, i], eval_set=[(X_te, y_te[:, i])], verbose=False)
        val_r2_list.append(r2_score(y_val_t[:, i], m.predict(X_tr)))
        models.append(m)

    # Evaluate on test set
    test_preds = np.column_stack([m.predict(X_te) for m in models])
    per_target = _per_target_r2(y_te, test_preds)
    overall = r2_score(y_te, test_preds)

    print(f"  Test R² (overall): {overall:.4f}")
    for k, v in per_target.items():
        print(f"    {k}: {v:.4f}")

    # Persist model
    out_dir = Path("results/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out_dir / "xgb_models.pkl")

    return per_target


# =============================================================================
# BASELINE B: VANILLA MLP (no physics)
# =============================================================================

def train_vanilla_mlp(
    epochs: int = 200,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict[str, float]:
    """
    Train an MLP with identical architecture as HINN but lambda=0 (no physics).

    Rationale: This ablates the physics constraint mechanism. HINN must produce
    a higher test R² AND a lower CVR than this model to justify the penalty.
    If the vanilla MLP achieves the same R², the constraint loss provides no
    predictive benefit (only a constraint satisfaction benefit, which is still
    publishable but weaker).
    """
    print(f"\n[Vanilla MLP] seed={seed}, epochs={epochs}")
    set_global_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test = _load_splits(seed)
    scaler_X, scaler_y_area, scaler_y_lat = _build_scalers(X_train, y_train)

    X_tr, y_tr = _transform(X_train, y_train, scaler_X, scaler_y_area, scaler_y_lat)
    X_v, y_v = _transform(X_val, y_val, scaler_X, scaler_y_area, scaler_y_lat)
    X_te, y_te = _transform(X_test, y_test, scaler_X, scaler_y_area, scaler_y_lat)

    def _make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    train_loader = _make_loader(X_tr, y_tr, shuffle=True)
    val_loader = _make_loader(X_v, y_v, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HINN_MultiTask(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(X_v)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4} | val_mse={val_loss:.4f}")

    # Evaluate best model on test set
    model.load_state_dict(best_state)
    model.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_preds = model(X_te_t).cpu().numpy()

    per_target = _per_target_r2(y_te, test_preds)
    overall = r2_score(y_te, test_preds)
    print(f"  Test R² (overall): {overall:.4f}")
    for k, v in per_target.items():
        print(f"    {k}: {v:.4f}")

    # Persist
    out_dir = Path("results/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "vanilla_mlp_best.pt")

    return per_target


# =============================================================================
# ENTRY POINT: run both and write comparison CSV
# =============================================================================

def train_baselines(epochs: int = 200, seed: int = 42) -> None:
    """Train XGBoost and Vanilla MLP, write unified results/models/baseline_results.csv."""
    rows: list[dict] = []

    xgb_r2 = train_xgboost(seed=seed)
    for target, r2 in xgb_r2.items():
        rows.append({"seed": seed, "model": "XGBoost", "split": "test", "target": target, "r2": r2})

    mlp_r2 = train_vanilla_mlp(epochs=epochs, seed=seed)
    for target, r2 in mlp_r2.items():
        rows.append({"seed": seed, "model": "VanillaMLP", "split": "test", "target": target, "r2": r2})

    out = Path("results/models/baseline_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nBaseline results saved to {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_baselines(epochs=args.epochs, seed=args.seed)
