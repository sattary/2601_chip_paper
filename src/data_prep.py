import re
import ast
import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_DATA_PATH = Path("data/db4hls.sql")
TARGET_COLS = ["hls_lut", "hls_ff", "average_latency", "best_latency"]

def parse_mysqldump(filepath):
    print(f"Parsing {filepath}...")
    tables = {}
    current_table = None
    table_schemas = {}
    
    re_create = re.compile(r"^CREATE TABLE `(\w+)` \(", re.IGNORECASE)
    re_insert = re.compile(r"^INSERT INTO `(\w+)` VALUES", re.IGNORECASE)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if match := re_create.match(line):
                current_table = match.group(1)
                table_schemas[current_table] = []
                tables[current_table] = []
                continue
                
            if current_table and line.startswith("`"):
                col_match = re.search(r"`(\w+)`", line)
                if col_match: table_schemas[current_table].append(col_match.group(1))
                continue
            
            if line.startswith(")") and current_table:
                current_table = None
                continue
                
            if match := re_insert.match(line):
                table_name = match.group(1)
                if table_name not in tables: tables[table_name] = []
                # Handle giant value blocks
                data = line[line.find('VALUES ')+7 : line.rfind(';')]
                # Split records carefully
                records = data.split('),(')
                for rec in records:
                    rec = rec.strip('()')
                    # Simple split, handles most tables except config
                    parts = [p.strip(" '") for p in rec.split(',')]
                    tables[table_name].append(parts)
                    
    # Convert to DataFrames
    for name in list(tables.keys()):
        cols = table_schemas.get(name)
        data = tables[name]
        if not data: 
            if name in tables: del tables[name]
            continue
        
        if cols:
            max_cols = len(cols)
            # Ensure each row has same length as columns
            clean_data = []
            for row in data:
                if len(row) > max_cols: clean_data.append(row[:max_cols])
                elif len(row) < max_cols: clean_data.append(row + [None]*(max_cols - len(row)))
                else: clean_data.append(row)
            tables[name] = pd.DataFrame(clean_data, columns=cols)
        else:
            tables[name] = pd.DataFrame(data)
        
    return tables

def main():
    tables = parse_mysqldump(DEFAULT_DATA_PATH)
    
    if 'implementation' not in tables:
        print("Error: implementation table not found.")
        return

    impl = tables['implementation']
    perf = tables.get('performance_results')
    res = tables.get('resource_results')
    conf = tables.get('configuration')
    
    # Cleaning types
    for df_tmp in [impl, perf, res]:
        if df_tmp is not None:
            for col in df_tmp.columns:
                if 'id' in col.lower() or any(x in col for x in ['clock', 'latency', 'interval', 'hls_']):
                    df_tmp[col] = pd.to_numeric(df_tmp[col], errors='coerce')

    print("Merging core tables...")
    # Fix relational key mismatches (plural vs singular)
    if perf is not None:
        if 'id_performance_results' in perf.columns and 'id_performance_results' in impl.columns:
            df = impl.merge(perf, on='id_performance_results')
        elif 'id_performance_result' in perf.columns:
            perf = perf.rename(columns={'id_performance_result': 'id_performance_results'})
            df = impl.merge(perf, on='id_performance_results')
        else:
            print("Warning: could not find matching performance keys.")
            df = impl

    if res is not None:
        if 'id_resource_result' in res.columns:
            res = res.rename(columns={'id_resource_result': 'id_resource_results'})
        if 'id_resource_results' in res.columns and 'id_resource_results' in df.columns:
            df = df.merge(res, on='id_resource_results')
    
    # Parse parameters
    if conf is not None:
        print("Parsing parameters...")
        def parse_p(p_str):
            try:
                p = ast.literal_eval(p_str.replace("\\'", "'").replace("\\n", " "))
                return [str(item) for item in p]
            except: return []

        conf['params'] = conf['config'].apply(parse_p)
        max_len = max(conf['params'].apply(len)) if not conf.empty else 0
        params_df = pd.DataFrame(conf['params'].tolist(), columns=[f"param_{i}" for i in range(max_len)], index=conf.index)
        conf_clean = pd.concat([conf[['hash_configuration']], params_df], axis=1)
        df = df.merge(conf_clean, on='hash_configuration')
    
    # Add kernel names from sidecar
    mapping_path = Path("data/processed/kernel_mapping.parquet")
    if mapping_path.exists():
        mapping = pd.read_parquet(mapping_path)
        df = df.merge(mapping, on='hash_configuration', how='left')
        print(f"Added kernel names. Unique: {df['kernel_name'].nunique() if 'kernel_name' in df.columns else 0}")
    
    print(f"Final dataset: {df.shape}")
    if not df.empty:
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_parquet("data/processed/master.parquet", index=False)
        
        # Targets
        y = df[TARGET_COLS + (['kernel_name'] if 'kernel_name' in df.columns else [])]
        
        # Features
        feature_cols = [c for c in df.columns if c.startswith("param_")]
        for syn in ['target_clock', 'estimated_clock', 'hls_dsp', 'hls_bram']:
            if syn in df.columns: feature_cols.append(syn)
            
        X = pd.get_dummies(df[feature_cols])
        X.to_parquet("data/processed/features.parquet", index=False)
        y.to_parquet("data/processed/targets.parquet", index=False)
        print("Saved processed files to data/processed/")
    else:
        print("Error: Merged dataframe is empty.")

if __name__ == "__main__":
    main()
