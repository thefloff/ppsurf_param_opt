#!/usr/bin/env python3
"""
Script to update set files (testset.txt, trainset.txt, valset.txt) 
by keeping only lines that correspond to valid files in the 06_opt_params directory.
"""

import os
from pathlib import Path


def main():
    # Define paths
    dataset_dir = Path("datasets/abc_normals")
    opt_params_dir = dataset_dir / "06_opt_params"
    
    # Set files to update
    set_files = ["testset.txt", "trainset.txt", "valset.txt"]
    
    # Get all valid filenames from 06_opt_params (without .json extension)
    print(f"Reading valid files from {opt_params_dir}")
    if not opt_params_dir.exists():
        print(f"Error: Directory {opt_params_dir} does not exist!")
        return
    
    valid_files = set()
    json_files = list(opt_params_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {opt_params_dir}")
    
    for json_file in json_files:
        # Remove .json extension to get the base name
        base_name = json_file.stem
        valid_files.add(base_name)
    
    print(f"Found {len(valid_files)} valid base names")
    
    # Update each set file
    for set_file in set_files:
        set_file_path = dataset_dir / set_file
        
        if not set_file_path.exists():
            print(f"Warning: {set_file_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {set_file_path}")
        
        # Read current lines
        with open(set_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        print(f"  Original file has {len(lines)} lines")
        
        # Filter lines to keep only valid ones
        valid_lines = []
        for line in lines:
            if line in valid_files:
                valid_lines.append(line)
        
        print(f"  Keeping {len(valid_lines)} valid lines")
        print(f"  Removing {len(lines) - len(valid_lines)} invalid lines")
        
        # Write back the filtered lines
        with open(set_file_path, 'w') as f:
            for line in valid_lines:
                f.write(line + '\n')
        
        print(f"  Updated {set_file_path}")
    
    print("\nSet file updates completed!")
    
    # Print summary
    print("\nSummary:")
    for set_file in set_files:
        set_file_path = dataset_dir / set_file
        if set_file_path.exists():
            with open(set_file_path, 'r') as f:
                line_count = len(f.readlines())
            print(f"  {set_file}: {line_count} lines")


if __name__ == "__main__":
    main()