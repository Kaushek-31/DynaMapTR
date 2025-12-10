# Save as: tools/debug_annotations.py
"""Debug why objects aren't being extracted."""
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)

# Check first sample
scene = nusc.scene[0]
sample = nusc.get('sample', scene['first_sample_token'])

print(f"\nSample token: {sample['token']}")
print(f"Number of annotations: {len(sample['anns'])}")

# Check what category names actually look like
for ann_token in sample['anns'][:5]:  # First 5
    ann = nusc.get('sample_annotation', ann_token)
    print(f"\nCategory: '{ann['category_name']}'")
    print(f"  Num lidar pts: {ann.get('num_lidar_pts', 0)}")
    print(f"  Num radar pts: {ann.get('num_radar_pts', 0)}")

# Expected class names
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

print(f"\n\nChecking class name matching:")
for ann_token in sample['anns'][:10]:
    ann = nusc.get('sample_annotation', ann_token)
    category = ann['category_name']
    
    # Check if it matches
    if category in class_names:
        print(f"  ✓ '{category}' - MATCH")
    else:
        # Check if it's a subcategory
        base_category = category.split('.')[0] if '.' in category else category
        if base_category in class_names:
            print(f"  ~ '{category}' -> '{base_category}' - PARTIAL MATCH")
        else:
            print(f"  ✗ '{category}' - NO MATCH")
