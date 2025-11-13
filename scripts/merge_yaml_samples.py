#!/usr/bin/env python3
"""
Script to merge two YAML sample files, incrementing sample IDs from the second file.
"""

import yaml
import sys
from pathlib import Path


def merge_yaml_samples(file1_path, file2_path, output_path):
    """
    Merge two YAML sample files, incrementing sample IDs from file2.

    Args:
        file1_path: Path to first YAML file
        file2_path: Path to second YAML file
        output_path: Path for merged output file
    """
    # Read first file
    print(f"Reading {file1_path}...")
    with open(file1_path, 'r') as f:
        data1 = yaml.safe_load(f)

    # Read second file
    print(f"Reading {file2_path}...")
    with open(file2_path, 'r') as f:
        data2 = yaml.safe_load(f)

    # Get samples from both files
    samples1 = data1['collected_samples']['samples']
    samples2 = data2['collected_samples']['samples']

    num_samples1 = len(samples1)
    num_samples2 = len(samples2)

    print(f"File 1 has {num_samples1} samples")
    print(f"File 2 has {num_samples2} samples")

    # Increment sample IDs in second file
    print(f"Incrementing sample IDs in file 2 by {num_samples1}...")
    for sample in samples2:
        sample['sample_id'] += num_samples1

    # Merge samples
    merged_samples = samples1 + samples2
    total_samples = len(merged_samples)

    # Create merged data structure
    merged_data = {
        'collected_samples': {
            'timestamp': data1['collected_samples']['timestamp'],
            'total_samples': total_samples,
            'samples': merged_samples
        }
    }

    # Write merged file
    print(f"Writing merged file to {output_path}...")
    print(f"Total samples: {total_samples}")
    with open(output_path, 'w') as f:
        yaml.dump(merged_data, f, default_flow_style=False, sort_keys=False)

    print("Done!")


if __name__ == "__main__":
    # Define file paths
    base_path = Path(__file__).parent.parent / "data" / "results"

    file1 = base_path / "collected_samples_500_lowpass.yaml"
    file2 = base_path / "collected_samples_250_lowpass_filter.yaml"
    output = base_path / "collected_samples_750_merged.yaml"

    merge_yaml_samples(file1, file2, output)
