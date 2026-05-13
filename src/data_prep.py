"""
Data Preparation Module for HLS Design Space Exploration.

This module parses MySQL dump files from db4hls, cleans the data,
merges tables, and prepares features for machine learning.
"""

from pathlib import Path

import pandas as pd
import re


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "db4hls.sql"

# Table names in the database
TABLE_IMPLEMENTATION = "implementation"
TABLE_RESOURCE_RESULTS = "resource_results"
TABLE_PERFORMANCE_RESULTS = "performance_results"
TABLE_CONFIGURATION = "configuration"

# Column names
COL_HASH_CONFIG = "hash_configuration"
COL_ID_RESOURCE = "id_resource_results"
COL_ID_PERFORMANCE = "id_performance_results"

# Target columns for ML (objectives to predict)
# Area objectives: hls_lut, hls_ff (correlated ~0.74)
# Timing objectives: average_latency, best_latency (conflicts with area)
TARGET_COLUMNS = [
    "hls_lut",  # Area: LUT usage
    "hls_ff",  # Area: Flip-flop usage
    "average_latency",  # Timing: Average latency (cycles)
    "best_latency",  # Timing: Best-case latency (cycles)
]

# Columns to drop (metadata not useful for ML)
DROP_COLUMNS = [
    "timestamp",
    "timeout",
    "hls_execution_time",
    "hls_exit_value",
    "hash_configuration",
    "platform",
]


# =============================================================================
# SQL PARSING (Memory-Efficient)
# =============================================================================


def parse_mysqldump(filepath: Path | str) -> dict[str, pd.DataFrame]:
    """
    Parse a MySQL dump file and extract tables as DataFrames.

    Uses memory-efficient streaming to handle large files.

    Args:
        filepath: Path to the .sql dump file.

    Returns:
        Dictionary mapping table names to pandas DataFrames.
    """
    import gc

    filepath = Path(filepath)

    # Regex patterns
    re_create = re.compile(r"CREATE TABLE `(\w+)` \(")
    re_column = re.compile(r"^\s*`(\w+)`")
    re_insert_header = re.compile(r"INSERT INTO `(\w+)` VALUES ")

    table_schemas: dict[str, list[str]] = {}
    table_data: dict[str, list[list[str]]] = {}
    current_table: str | None = None

    print(f"Parsing {filepath}...")

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            # Capture CREATE TABLE statements
            if match := re_create.match(line):
                current_table = match.group(1)
                table_schemas[current_table] = []
                table_data[current_table] = []
                print(f"  Found table: {current_table}")
                continue

            # Capture column definitions within CREATE block
            if current_table and line.startswith("`"):
                if match := re_column.match(line):
                    table_schemas[current_table].append(match.group(1))

            # End of CREATE block
            if line.endswith(";") and current_table:
                current_table = None

            # Capture INSERT statements (memory-efficient streaming)
            if match := re_insert_header.match(line):
                table_name = match.group(1)

                if table_name in table_schemas:
                    # Get the VALUES portion (skip header, remove trailing semicolon)
                    values_start = match.end()
                    values_str = (
                        line[values_start:-1]
                        if line.endswith(";")
                        else line[values_start:]
                    )

                    # Stream parse rows without creating huge intermediate list
                    rows = _parse_values_streaming(values_str)
                    table_data[table_name].extend(rows)

                    print(
                        f"    Line {line_num}: loaded {len(rows)} rows into {table_name}"
                    )

                    # Force garbage collection after large inserts
                    del values_str, rows
                    gc.collect()

    # Convert to DataFrames
    print("Converting to DataFrames...")
    return _build_dataframes(table_schemas, table_data)


def _parse_values_streaming(values_str: str) -> list[list[str]]:
    """
    Parse VALUES string using streaming approach.

    Handles: (val1, 'text', val3), (val4, 'text', val6), ...
    """
    rows = []
    current_row: list[str] = []
    current_value: list[str] = []
    in_quotes = False
    depth = 0

    i = 0
    n = len(values_str)

    while i < n:
        char = values_str[i]

        if char == "'" and (i == 0 or values_str[i - 1] != "\\"):
            # Toggle quote state (but don't include the quote in value)
            in_quotes = not in_quotes
        elif char == "(" and not in_quotes:
            depth += 1
            if depth == 1:
                # Start of a new row
                current_row = []
                current_value = []
        elif char == ")" and not in_quotes:
            depth -= 1
            if depth == 0:
                # End of row - save last value and row
                if current_value:
                    current_row.append("".join(current_value).strip())
                rows.append(current_row)
                current_row = []
                current_value = []
        elif char == "," and not in_quotes and depth == 1:
            # Value separator within row
            current_row.append("".join(current_value).strip())
            current_value = []
        elif depth >= 1:
            # Part of a value
            current_value.append(char)

        i += 1

    return rows


def _build_dataframes(
    schemas: dict[str, list[str]],
    data: dict[str, list[list[str]]],
) -> dict[str, pd.DataFrame]:
    """Convert parsed data to pandas DataFrames."""
    import gc

    tables = {}

    for table_name, rows in data.items():
        columns = schemas.get(table_name, [])

        if not rows:
            tables[table_name] = pd.DataFrame(columns=columns)
        elif len(rows[0]) == len(columns):
            tables[table_name] = pd.DataFrame(rows, columns=columns)
        else:
            print(
                f"  Warning: {table_name} has {len(columns)} cols but data has {len(rows[0])}"
            )
            tables[table_name] = pd.DataFrame(rows)

        print(f"  -> {table_name}: {len(tables[table_name])} rows")
        gc.collect()

    return tables


# =============================================================================
# DATA CLEANING
# =============================================================================


def clean_tables(db: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Clean and prepare individual tables for merging.

    Args:
        db: Dictionary of raw DataFrames from parse_mysqldump.

    Returns:
        Dictionary of cleaned DataFrames.
    """
    cleaned = {}

    cleaned["implementation"] = _clean_implementation(db[TABLE_IMPLEMENTATION].copy())
    cleaned["resource_results"] = _clean_results_table(
        db[TABLE_RESOURCE_RESULTS].copy(), "id", COL_ID_RESOURCE
    )
    cleaned["performance_results"] = _clean_results_table(
        db[TABLE_PERFORMANCE_RESULTS].copy(), "id", COL_ID_PERFORMANCE
    )
    cleaned["configuration"] = _clean_configuration(db[TABLE_CONFIGURATION].copy())

    return cleaned


def _clean_implementation(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the implementation table."""
    # Convert ID columns to numeric
    for col in [COL_ID_RESOURCE, COL_ID_PERFORMANCE]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid IDs
    df.dropna(subset=[COL_ID_RESOURCE, COL_ID_PERFORMANCE], inplace=True)

    # Cast to integer
    df[COL_ID_RESOURCE] = df[COL_ID_RESOURCE].astype(int)
    df[COL_ID_PERFORMANCE] = df[COL_ID_PERFORMANCE].astype(int)

    return df


def _clean_results_table(
    df: pd.DataFrame,
    id_col: str,
    alt_id_col: str,
) -> pd.DataFrame:
    """Clean a results table and set its index."""
    target_col = id_col if id_col in df.columns else alt_id_col

    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.set_index(target_col)

    return df


def _clean_configuration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the configuration table and parse config lists into feature columns.

    The 'config' column contains stringified Python lists that need to be
    parsed into individual parameter columns.
    """
    import ast

    # Rename hash column (typically at index 1)
    if 1 in df.columns:
        df.rename(columns={1: COL_HASH_CONFIG}, inplace=True)

    # Drop useless first column if present
    if 0 in df.columns:
        df.drop(columns=[0], inplace=True)

    # Remove empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Find the config column (contains the stringified list)
    config_col = None
    for col in df.columns:
        if col != COL_HASH_CONFIG:
            sample = str(df[col].iloc[0]) if len(df) > 0 else ""
            if sample.startswith("[") and "'" in sample:
                config_col = col
                break

    # Parse the config column into individual features
    if config_col is not None:
        print(f"  Parsing '{config_col}' column into individual features...")
        parsed_rows = []

        for idx, val in enumerate(df[config_col]):
            try:
                # Fix escaped quotes and parse as Python list
                clean_val = str(val).replace("\\'", "'").replace("\\n", " ")
                parsed = ast.literal_eval(clean_val)

                # Flatten tuples within the list
                flat = []
                for item in parsed:
                    if isinstance(item, (list, tuple)):
                        flat.extend(str(x) for x in item)
                    else:
                        flat.append(str(item))
                parsed_rows.append(flat)
            except (ValueError, SyntaxError):
                # If parsing fails, use empty list
                parsed_rows.append([])

        # Find max length for padding
        max_len = max(len(row) for row in parsed_rows) if parsed_rows else 0

        # Pad shorter rows
        for row in parsed_rows:
            while len(row) < max_len:
                row.append(None)

        # Create DataFrame from parsed values
        param_cols = [f"param_{i}" for i in range(max_len)]
        df_params = pd.DataFrame(parsed_rows, columns=param_cols, index=df.index)

        # Drop original config columns and join parsed parameters
        cols_to_drop = [c for c in df.columns if c != COL_HASH_CONFIG]
        df = df.drop(columns=cols_to_drop)
        df = df.join(df_params)

        print(f"  Extracted {max_len} parameter columns")
    else:
        # Fallback: just rename remaining columns
        rename_map = {
            col: f"config_{col}" for col in df.columns if col != COL_HASH_CONFIG
        }
        df.rename(columns=rename_map, inplace=True)

    return df


# =============================================================================
# TABLE MERGING
# =============================================================================


def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all cleaned tables into a single master DataFrame.

    Args:
        tables: Dictionary of cleaned DataFrames.

    Returns:
        Merged master DataFrame.
    """
    df = tables["implementation"]

    # Join with resource results
    df = df.join(
        tables["resource_results"],
        on=COL_ID_RESOURCE,
        rsuffix="_res",
    )

    # Join with performance results
    df = df.join(
        tables["performance_results"],
        on=COL_ID_PERFORMANCE,
        rsuffix="_perf",
    )

    # Merge configuration on hash
    df[COL_HASH_CONFIG] = df[COL_HASH_CONFIG].astype(str)
    config = tables["configuration"].copy()
    config[COL_HASH_CONFIG] = config[COL_HASH_CONFIG].astype(str)

    df = df.merge(config, on=COL_HASH_CONFIG, how="left")

    return df


def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final cleaning steps to the merged dataset.

    Args:
        df: Merged master DataFrame.

    Returns:
        Finalized DataFrame ready for ML.
    """
    # Convert columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Keep as-is if conversion fails

    # Drop metadata columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Drop rows where synthesis failed (missing LUT count)
    if "lut" in df.columns:
        df.dropna(subset=["lut"], inplace=True)

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def prepare_ml_data(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature matrix X and target matrix y for machine learning.

    Args:
        df: Finalized master DataFrame.
        target_cols: List of target column names. Defaults to TARGET_COLUMNS.

    Returns:
        Tuple of (X, y) DataFrames.
    """
    if target_cols is None:
        target_cols = TARGET_COLUMNS

    # Filter to existing target columns
    targets = [c for c in target_cols if c in df.columns]

    # Get feature columns (param_* and config_* columns)
    feature_cols = [c for c in df.columns if c.startswith(("param_", "config_"))]

    X = df[feature_cols].copy()
    y = df[targets].copy()

    # Handle missing values and encode
    X = _encode_features(X)

    # Drop rows with missing targets
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    return X, y


def _encode_features(df: pd.DataFrame, max_categories: int = 50) -> pd.DataFrame:
    """
    Encode features: fill NaNs and one-hot encode low-cardinality categoricals.

    Args:
        df: Feature DataFrame.
        max_categories: Maximum unique values for one-hot encoding.
                       Columns with more categories use label encoding.
    """
    from sklearn.preprocessing import LabelEncoder

    num_cols = []
    cat_cols_low_cardinality = []
    cat_cols_high_cardinality = []

    for col in df.columns:
        try:
            pd.to_numeric(df[col].dropna().unique())
            num_cols.append(col)
        except (ValueError, TypeError):
            n_unique = df[col].nunique()
            if n_unique <= max_categories:
                cat_cols_low_cardinality.append(col)
            else:
                cat_cols_high_cardinality.append(col)
                print(f"  Skipping one-hot for '{col}' ({n_unique} unique values)")

    # Fill missing values
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)
    if cat_cols_low_cardinality:
        df[cat_cols_low_cardinality] = df[cat_cols_low_cardinality].fillna("None")
    if cat_cols_high_cardinality:
        df[cat_cols_high_cardinality] = df[cat_cols_high_cardinality].fillna("None")

    # One-hot encode low-cardinality categorical columns
    if cat_cols_low_cardinality:
        df = pd.get_dummies(df, columns=cat_cols_low_cardinality, drop_first=True)

    # Label encode high-cardinality columns (or just drop them)
    for col in cat_cols_high_cardinality:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def load_and_prepare(
    filepath: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: load SQL dump, clean, merge, and prepare for ML.

    Args:
        filepath: Path to the SQL dump file. Defaults to DEFAULT_DATA_PATH.

    Returns:
        Tuple of (master_df, X, y).
    """
    if filepath is None:
        filepath = DEFAULT_DATA_PATH

    # Parse SQL dump
    raw_tables = parse_mysqldump(filepath)

    # Clean tables
    clean = clean_tables(raw_tables)

    # Merge into master dataset
    master = merge_tables(clean)
    master = finalize_dataset(master)

    # Prepare ML data
    X, y = prepare_ml_data(master)

    return master, X, y


# =============================================================================
# DATA PERSISTENCE
# =============================================================================


def save_to_parquet(df: pd.DataFrame, filepath: Path | str) -> None:
    """Save DataFrame to Parquet format (fast, compressed)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    print(f"Saved to {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")


def save_to_csv(df: pd.DataFrame, filepath: Path | str) -> None:
    """Save DataFrame to CSV format (human-readable)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")


def save_ml_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    output_dir: Path | str,
    format: str = "parquet",
) -> None:
    """
    Save feature and target matrices for ML.

    Args:
        X: Feature matrix.
        y: Target matrix.
        output_dir: Directory to save files.
        format: 'parquet' or 'csv'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_fn = save_to_parquet if format == "parquet" else save_to_csv
    ext = "parquet" if format == "parquet" else "csv"

    save_fn(X, output_dir / f"features.{ext}")
    save_fn(y, output_dir / f"targets.{ext}")


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================


def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize features using StandardScaler."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_scaled


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with zero variance."""
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"  Removing {len(constant_cols)} constant columns")
        df = df.drop(columns=constant_cols)
    return df


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare HLS dataset for ML")
    parser.add_argument(
        "--input", "-i", type=str, default=None, help="Path to SQL dump"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format",
    )
    parser.add_argument(
        "--normalize", "-n", action="store_true", help="Normalize features"
    )
    args = parser.parse_args()

    print("Loading and preparing HLS dataset...")
    master_df, X, y = load_and_prepare(args.input)

    # Remove constant columns
    X = remove_constant_columns(X)

    # Optional normalization
    if args.normalize:
        print("Normalizing features...")
        X = normalize_features(X)

    print(f"\n{'=' * 50}")
    print(f"Master DataFrame: {master_df.shape}")
    print(f"Features (X):     {X.shape}")
    print(f"Targets (y):      {y.shape}")
    print(f"{'=' * 50}")
    print(f"\nTarget columns: {list(y.columns)}")
    print(f"\nFeature columns: {list(X.columns)}")

    # Save outputs
    print(f"\nSaving to {args.output}/...")
    save_ml_data(X, y, args.output, format=args.format)
    save_to_parquet(master_df, Path(args.output) / f"master.{args.format}")

    print("\nDone!")
