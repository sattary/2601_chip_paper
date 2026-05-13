"""
Monotonicity Ground Truth Audit.

Rationale:
The HINN physics penalty assumes that increasing param_3 (loop unroll / parallelism)
monotonically increases area AND decreases latency. This script empirically audits
that assumption in the raw db4hls ground truth data.

Result (from preliminary sampling): the joint monotonicity holds in ~52% of
consecutive pairs — near-random. This is expected: db4hls spans heterogeneous
kernels (FFT, GEMM, conv, etc.) where the global monotonicity is confounded by
cross-kernel variation, memory bandwidth bottlenecks, and tool heuristics.

The script produces a per-group audit CSV so the paper can characterize which
kernel sub-spaces have clean monotone behaviour (the valid domain of the prior).
"""

import itertools
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def run_monotonicity_audit(
    master_parquet: str = "data/processed/master.parquet",
    output_csv: str = "results/data/monotonicity_audit.csv",
    max_groups: Optional[int] = None,
) -> pd.DataFrame:
    """
    Audit the ground truth monotonicity of param_3 (parallelism factor).

    For each group of rows sharing the same values of all parameters EXCEPT
    param_3, this function sorts by numeric param_3 and checks whether
    consecutive rows satisfy:
        area(i+1)   >= area(i)     [monotone increasing]
        latency(i+1) <= latency(i) [monotone decreasing]

    Args:
        master_parquet: Path to master.parquet (pre-feature-encoding dataset).
        output_csv:     Path to write per-group audit results.
        max_groups:     Limit group iteration (for quick debugging). None = all.

    Returns:
        DataFrame with summary statistics per group.
    """
    print(f"Loading {master_parquet}...")
    master = pd.read_parquet(master_parquet)

    # Filter to rows with numeric param_3 and valid targets
    master["param_3_num"] = pd.to_numeric(master["param_3"], errors="coerce")
    mc = master.dropna(subset=["param_3_num", "hls_lut", "average_latency"]).copy()
    mc["hls_lut"] = pd.to_numeric(mc["hls_lut"], errors="coerce")
    mc["average_latency"] = pd.to_numeric(mc["average_latency"], errors="coerce")
    mc = mc.dropna(subset=["hls_lut", "average_latency"])

    print(f"Rows with numeric param_3 and valid targets: {len(mc)}")

    # Group by all parameters EXCEPT param_3 (the dimension being perturbed)
    group_cols = [
        c for c in [
            "param_0", "param_1", "param_2",
            "param_4", "param_5", "param_6", "param_7",
            "param_8", "param_9", "param_10",
        ]
        if c in mc.columns
    ]

    groups = mc.groupby(group_cols, dropna=False)
    print(f"Total groups: {len(groups)}")

    records = []
    total_pairs = 0
    both_mono = 0
    area_only = 0
    lat_only = 0
    neither = 0

    group_iter = itertools.islice(groups, max_groups) if max_groups else groups

    for group_key, g in group_iter:
        if len(g) < 2:
            continue

        g_sorted = g.sort_values("param_3_num")
        p3_values = g_sorted["param_3_num"].tolist()
        lut_values = g_sorted["hls_lut"].tolist()
        lat_values = g_sorted["average_latency"].tolist()

        grp_both = grp_area = grp_lat = grp_neither = 0

        for i in range(len(g_sorted) - 1):
            area_mono = lut_values[i + 1] >= lut_values[i]
            lat_mono = lat_values[i + 1] <= lat_values[i]
            total_pairs += 1

            if area_mono and lat_mono:
                both_mono += 1
                grp_both += 1
            elif area_mono:
                area_only += 1
                grp_area += 1
            elif lat_mono:
                lat_only += 1
                grp_lat += 1
            else:
                neither += 1
                grp_neither += 1

        grp_pairs = grp_both + grp_area + grp_lat + grp_neither
        records.append({
            "group_key": str(group_key),
            "n_configs": len(g_sorted),
            "param_3_values": str(p3_values),
            "pairs_total": grp_pairs,
            "pairs_both_mono": grp_both,
            "pairs_area_only": grp_area,
            "pairs_lat_only": grp_lat,
            "pairs_neither": grp_neither,
            "pct_both_mono": 100 * grp_both / grp_pairs if grp_pairs > 0 else float("nan"),
        })

    audit_df = pd.DataFrame(records)

    print(f"\n{'=' * 60}")
    print(f"Monotonicity Audit Summary ({total_pairs} consecutive pairs)")
    print(f"{'=' * 60}")
    print(f"  Both hold (area↑ AND latency↓): {both_mono:>6} ({100*both_mono/total_pairs:.1f}%)")
    print(f"  Area only (area↑, latency↗):    {area_only:>6} ({100*area_only/total_pairs:.1f}%)")
    print(f"  Latency only (area↓, latency↓): {lat_only:>6} ({100*lat_only/total_pairs:.1f}%)")
    print(f"  Neither:                         {neither:>6} ({100*neither/total_pairs:.1f}%)")
    print(f"{'=' * 60}")
    print(
        f"\nConclusion: joint monotonicity holds in {100*both_mono/total_pairs:.1f}% of pairs.\n"
        f"The HINN penalty acts as a SOFT REGULARIZER (Bayesian prior), not a hard constraint.\n"
        f"Include this analysis in Section 4.1 of the manuscript."
    )

    # Persist
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    print(f"\nPer-group audit saved to {out_path}")

    return audit_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit ground truth monotonicity in db4hls.")
    parser.add_argument("--master", default="data/processed/master.parquet")
    parser.add_argument("--out", default="results/data/monotonicity_audit.csv")
    parser.add_argument("--max-groups", type=int, default=None)
    args = parser.parse_args()

    run_monotonicity_audit(args.master, args.out, args.max_groups)
