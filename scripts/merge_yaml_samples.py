#!/usr/bin/env python3

import yaml
import sys
from pathlib import Path


def merge_yaml_samples(file1_path, file2_path, output_path):
    print(f"Reading {file1_path}...")
    with open(file1_path, 'r') as f:
        data1 = yaml.safe_load(f)

    print(f"Reading {file2_path}...")
    with open(file2_path, 'r') as f:
        data2 = yaml.safe_load(f)

    samples1 = data1['collected_samples']['samples']
    samples2 = data2['collected_samples']['samples']

    num_samples1 = len(samples1)
    num_samples2 = len(samples2)

    print(f"File 1 has {num_samples1} samples")
    print(f"File 2 has {num_samples2} samples")

    print(f"Incrementing sample IDs in file 2 by {num_samples1}...")
    for sample in samples2:
        sample['sample_id'] += num_samples1

    merged_samples = samples1 + samples2
    total_samples = len(merged_samples)

    merged_data = {
        'collected_samples': {
            'timestamp': data1['collected_samples']['timestamp'],
            'total_samples': total_samples,
            'samples': merged_samples
        }
    }

    print(f"Writing merged file to {output_path}...")
    print(f"Total samples: {total_samples}")
    with open(output_path, 'w') as f:
        yaml.dump(merged_data, f, default_flow_style=False, sort_keys=False)

    print("Done!")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "data" / "results"

    file1 = base_path / "collected_samples_500_lowpass.yaml"
    file2 = base_path / "collected_samples_250_lowpass_filter.yaml"
    output = base_path / "collected_samples_750_merged.yaml"

    merge_yaml_samples(file1, file2, output)
