"""
Pre-compute object annotations and save to PKL files.
This avoids loading objects on-the-fly during training.
"""

import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm
import argparse
import os


def add_object_annotations_to_infos(nusc, data_infos):
    """
    Add object annotations to each sample in data_infos.
    
    Args:
        nusc: NuScenes instance
        data_infos: List of sample info dicts
    
    Returns:
        Updated data_infos with 'object_annotations' added
    """
    print(f"Processing {len(data_infos)} samples...")
    
    for idx, info in enumerate(tqdm(data_infos)):
        try:
            sample_token = info['token']
            sample = nusc.get('sample', sample_token)
            
            annotations = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                
                # Skip objects with no LiDAR points
                if ann.get('num_lidar_pts', 0) == 0:
                    continue
                
                # Create box
                box = Box(
                    ann['translation'], 
                    ann['size'], 
                    Quaternion(ann['rotation'])
                )
                
                # Transform to lidar frame
                lidar_token = sample['data']['LIDAR_TOP']
                lidar_data = nusc.get('sample_data', lidar_token)
                cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
                
                # Global to ego
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                
                # Ego to lidar
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                # Store annotation
                annotations.append({
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),  # [width, length, height]
                    'rotation': [box.orientation.w, box.orientation.x, 
                               box.orientation.y, box.orientation.z],
                    'category_name': ann['category_name'],
                    'num_lidar_pts': ann.get('num_lidar_pts', 0),
                    'num_radar_pts': ann.get('num_radar_pts', 0),
                    'token': ann_token,
                })
            
            # Add to info dict
            info['object_annotations'] = annotations
            
        except Exception as e:
            print(f"Error processing sample {idx} ({info.get('token', 'unknown')}): {e}")
            info['object_annotations'] = []
    
    return data_infos


def main():
    parser = argparse.ArgumentParser(description='Create NuScenes infos with object annotations')
    parser.add_argument('--root-path', type=str, default='./data/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', 
                       choices=['v1.0-trainval', 'v1.0-test', 'v1.0-mini'])
    parser.add_argument('--out-dir', type=str, default='./data/nuscenes')
    args = parser.parse_args()
    
    # Initialize NuScenes
    print(f"Loading NuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.root_path, verbose=True)
    
    # Process train split
    train_pkl = os.path.join(args.root_path, 'nuscenes_map_infos_temporal_train.pkl')
    if os.path.exists(train_pkl):
        print(f"\n{'='*60}")
        print(f"Processing TRAIN split")
        print(f"{'='*60}")
        
        with open(train_pkl, 'rb') as f:
            train_infos = pickle.load(f)
        
        print(f"Loaded {len(train_infos['infos'])} train samples")
        
        # Add object annotations
        train_infos['infos'] = add_object_annotations_to_infos(nusc, train_infos['infos'])
        
        # Save updated PKL
        out_path = os.path.join(args.out_dir, 'nuscenes_map_infos_temporal_train_with_objects.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(train_infos, f)
        
        print(f"\n✓ Saved to: {out_path}")
        
        # Print statistics
        total_objects = sum(len(info['object_annotations']) for info in train_infos['infos'])
        avg_objects = total_objects / len(train_infos['infos'])
        print(f"Statistics: {total_objects} total objects, {avg_objects:.1f} avg per sample")
    
    # Process val split
    val_pkl = os.path.join(args.root_path, 'nuscenes_map_infos_temporal_val.pkl')
    if os.path.exists(val_pkl):
        print(f"\n{'='*60}")
        print(f"Processing VAL split")
        print(f"{'='*60}")
        
        with open(val_pkl, 'rb') as f:
            val_infos = pickle.load(f)
        
        print(f"Loaded {len(val_infos['infos'])} val samples")
        
        # Add object annotations
        val_infos['infos'] = add_object_annotations_to_infos(nusc, val_infos['infos'])
        
        # Save updated PKL
        out_path = os.path.join(args.out_dir, 'nuscenes_map_infos_temporal_val_with_objects.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(val_infos, f)
        
        print(f"\n✓ Saved to: {out_path}")
        
        # Print statistics
        total_objects = sum(len(info['object_annotations']) for info in val_infos['infos'])
        avg_objects = total_objects / len(val_infos['infos'])
        print(f"Statistics: {total_objects} total objects, {avg_objects:.1f} avg per sample")
    
    print(f"\n{'='*60}")
    print("✓ ALL DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()