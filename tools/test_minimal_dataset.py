# Save as: tools/test_dataset_minimal.py
"""Test dataset with minimal configuration."""
import sys
sys.path.insert(0, '.')

import pickle
import numpy as np

# Check PKL file directly
print("Loading PKL file...")
with open('./data/nuscenes/nuscenes_map_infos_temporal_train_mini_combined.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"✓ PKL loaded: {len(data['infos'])} samples")

info = data['infos'][0]
print("\nFirst sample info keys:")
for key in sorted(info.keys()):
    value = info[key]
    if isinstance(value, np.ndarray):
        print(f"  {key}: array {value.shape} {value.dtype}")
    elif isinstance(value, (list, tuple)) and len(value) > 0:
        if isinstance(value[0], np.ndarray):
            print(f"  {key}: list of {len(value)} arrays")
        else:
            print(f"  {key}: {type(value).__name__} len={len(value)}")
    elif isinstance(value, dict):
        print(f"  {key}: dict with keys {list(value.keys())}")
    else:
        print(f"  {key}: {type(value).__name__}")

print("\nChecking camera paths exist...")
import os
for cam_name, cam_info in info['cams'].items():
    path = cam_info['data_path']
    exists = os.path.exists(path)
    print(f"  {cam_name}: {path}")
    print(f"    Exists: {exists}")
    if not exists:
        print(f"    ✗ FILE NOT FOUND!")
        break