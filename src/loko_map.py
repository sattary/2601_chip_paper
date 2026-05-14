import pandas as pd
import re
from pathlib import Path

def main():
    print("Recovering kernel mapping from SQL (Two-pass method)...")
    hash_to_kernel = {}
    design_to_kernel = {} 
    space_to_design = {} 
    
    # PASS 1: Metadata
    with open('data/db4hls.sql', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_up = line.upper()
            if 'INSERT INTO `DESIGN` VALUES' in line_up or 'INSERT INTO DESIGN VALUES' in line_up:
                matches = re.findall(r"\('([^']*)',(\d+),", line)
                for name, id_d in matches: design_to_kernel[id_d] = name
            
            elif 'INSERT INTO `CONFIGURATION_SPACE` VALUES' in line_up or 'INSERT INTO CONFIGURATION_SPACE VALUES' in line_up:
                data = line[line.find('VALUES ')+7 : line.rfind(';')]
                for rec in data.split('),('):
                    rec = rec.strip('()')
                    parts = rec.split(',')
                    if len(parts) >= 3:
                        id_cs = parts[-3].strip()
                        id_d = parts[-2].strip()
                        if id_cs.isdigit() and id_d.isdigit():
                            space_to_design[id_cs] = id_d

    print(f"  Found {len(design_to_kernel)} designs.")
    print(f"  Found {len(space_to_design)} space mappings.")
    
    # PASS 2: Configurations
    with open('data/db4hls.sql', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_up = line.upper()
            if 'INSERT INTO `CONFIGURATION` VALUES' in line_up or 'INSERT INTO CONFIGURATION VALUES' in line_up:
                data = line[line.find('VALUES ')+7 : line.rfind(';')]
                for rec in data.split('),('):
                    rec = rec.strip('()')
                    parts = rec.split(',')
                    if len(parts) >= 2:
                        id_cs = parts[0].strip()
                        h = parts[1].strip("'")
                        if id_cs in space_to_design:
                            id_d = space_to_design[id_cs]
                            if id_d in design_to_kernel:
                                hash_to_kernel[h] = design_to_kernel[id_d]

    print(f"Mapped {len(hash_to_kernel)} unique hashes to kernels.")
    if hash_to_kernel:
        mapping_df = pd.DataFrame(list(hash_to_kernel.items()), columns=['hash_configuration', 'kernel_name'])
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        mapping_df.to_parquet('data/processed/kernel_mapping.parquet')
        print("Saved kernel mapping to data/processed/kernel_mapping.parquet")

if __name__ == "__main__":
    main()
