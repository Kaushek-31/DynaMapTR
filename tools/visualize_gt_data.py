"""
Visualize ground truth data from the dataset to verify correctness.
Saves BEV visualizations of:
- Map segmentation masks
- Object segmentation masks  
- 3D bounding boxes
- Map vectors
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon
import cv2
from pathlib import Path
import torch

from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin


def visualize_sample(dataset, idx, output_dir):
    """Visualize a single sample's ground truth data."""
    
    print(f"\n{'='*60}")
    print(f"Visualizing sample {idx}")
    print(f"{'='*60}")
    
    # Get data
    data = dataset[idx]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract info - FIXED
    img_metas = data['img_metas'].data if hasattr(data['img_metas'], 'data') else data['img_metas']
    
    # Handle both list and dict formats
    if isinstance(img_metas, list):
        sample_token = img_metas[0].get('sample_idx', f"sample_{idx}")
    elif isinstance(img_metas, dict):
        sample_token = img_metas.get('sample_idx', f"sample_{idx}")
    else:
        sample_token = f"sample_{idx}"
    
    print(f"Sample token: {sample_token}")
    print(f"img_metas type: {type(img_metas)}")
    print(f"Data keys: {list(data.keys())}")
    
    # ============================================================
    # 1. Visualize Map Segmentation Mask
    # ============================================================
    if 'gt_map_seg_mask' in data:
        gt_map_seg = data['gt_map_seg_mask'].data if hasattr(data['gt_map_seg_mask'], 'data') else data['gt_map_seg_mask']
        if isinstance(gt_map_seg, torch.Tensor):
            gt_map_seg = gt_map_seg.cpu().numpy()
        
        print(f"\n--- Map Segmentation ---")
        print(f"Shape: {gt_map_seg.shape}")
        print(f"Dtype: {gt_map_seg.dtype}")
        print(f"Unique values: {np.unique(gt_map_seg)}")
        print(f"Value range: [{gt_map_seg.min()}, {gt_map_seg.max()}]")
        
        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Handle different shapes
        if len(gt_map_seg.shape) == 3:
            # (C, H, W) or (1, H, W)
            if gt_map_seg.shape[0] == 1:
                map_viz = gt_map_seg[0]
            else:
                # Multi-channel: combine or show first
                map_viz = gt_map_seg.max(axis=0)
        else:
            map_viz = gt_map_seg
        
        im = ax.imshow(map_viz, cmap='tab20', interpolation='nearest')
        ax.set_title(f'Map Segmentation Mask\nShape: {gt_map_seg.shape}')
        ax.set_xlabel('X (BEV width)')
        ax.set_ylabel('Y (BEV height)')
        plt.colorbar(im, ax=ax, label='Class ID')
        
        save_path = output_dir / f'sample_{idx:04d}_map_seg.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    else:
        print("\n⚠ gt_map_seg_mask not found in data")
    
    # ============================================================
    # 2. Visualize Object Segmentation Mask
    # ============================================================
    if 'gt_object_seg_mask' in data and 'gt_bboxes_3d' in data:
        gt_obj_seg = data['gt_object_seg_mask'].data
        if isinstance(gt_obj_seg, torch.Tensor):
            gt_obj_seg = gt_obj_seg.cpu().numpy()
        if len(gt_obj_seg.shape) == 3:
            obj_viz = gt_obj_seg[0]
        else:
            obj_viz = gt_obj_seg
        
        gt_vecs = data['gt_bboxes_3d'].data if hasattr(data['gt_bboxes_3d'], 'data') else data['gt_bboxes_3d']
        gt_labels = data['gt_labels_3d'].data if hasattr(data['gt_labels_3d'], 'data') else data['gt_labels_3d']
        
        if hasattr(gt_vecs, 'instance_list') and len(gt_vecs.instance_list) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            
            # Show object mask as background
            ax.imshow(obj_viz, cmap='gray', alpha=0.5, extent=[-30, 30, -15, 15], origin='lower')
            
            # Overlay map vectors
            for instance, label in zip(gt_vecs.instance_list, gt_labels):
                coords = np.array(list(instance.coords))[:, :2]
                color = ['red', 'green', 'blue'][int(label) % 3]
                ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
            
            ax.set_xlim(-30, 30)
            ax.set_ylim(-15, 15)
            ax.set_title('Coordinate Alignment Check\n(Objects should align with map)')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            
            save_path = output_dir / f'sample_{idx:04d}_alignment_check.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved alignment check: {save_path}")
            plt.close()
    else:
        print("\n⚠ gt_object_seg_mask not found in data")
    
    # ============================================================
    # 3. Visualize Map Vectors (if using MapTR)
    # ============================================================
    if 'gt_bboxes_3d' in data and 'gt_labels_3d' in data:
        gt_vecs = data['gt_bboxes_3d'].data if hasattr(data['gt_bboxes_3d'], 'data') else data['gt_bboxes_3d']
        gt_labels = data['gt_labels_3d'].data if hasattr(data['gt_labels_3d'], 'data') else data['gt_labels_3d']
        
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()
        
        print(f"\n--- Map Vectors ---")
        print(f"Type: {type(gt_vecs)}")
        print(f"Labels shape: {gt_labels.shape if hasattr(gt_labels, 'shape') else type(gt_labels)}")
        
        # Check if it's LiDARInstanceLines
        if hasattr(gt_vecs, 'instance_list'):
            num_instances = len(gt_vecs.instance_list)
            print(f"Number of map instances: {num_instances}")
            
            if num_instances > 0:
                fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                ax.set_aspect('equal')
                
                # Plot each vector
                colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
                class_names = ['divider', 'ped_crossing', 'boundary']
                
                label_counts = {}
                for instance, label in zip(gt_vecs.instance_list, gt_labels):
                    coords = np.array(list(instance.coords))
                    if coords.shape[1] > 2:
                        coords = coords[:, :2]
                    color = colors[int(label) % len(colors)]
                    label_name = class_names[int(label)] if int(label) < len(class_names) else f'class_{label}'
                    
                    # Track label for legend
                    if label_name not in label_counts:
                        label_counts[label_name] = 0
                        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2, label=label_name)
                    else:
                        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
                    
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
                
                ax.legend()
                
                # Print counts
                print(f"Vector counts by class:")
                for name, count in label_counts.items():
                    print(f"  {name}: {count}")
                
                ax.set_xlim(-30, 30)
                ax.set_ylim(-15, 15)
                ax.set_xlabel('X (meters)')
                ax.set_ylabel('Y (meters)')
                ax.set_title(f'Map Vectors (Ground Truth)\n{num_instances} instances')
                ax.grid(True, alpha=0.3)
                
                save_path = output_dir / f'sample_{idx:04d}_map_vectors.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {save_path}")
                plt.close()
            else:
                print("  No map instances to visualize")
        else:
            print("  gt_bboxes_3d doesn't have instance_list attribute")
    else:
        print("\n⚠ gt_bboxes_3d or gt_labels_3d not found in data")
    
    # ============================================================
    # 4. Combined Visualization
    # ============================================================
    has_map = 'gt_map_seg_mask' in data
    has_obj = 'gt_object_seg_mask' in data
    
    if has_map or has_obj:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Map seg
        if has_map:
            gt_map_seg = data['gt_map_seg_mask'].data if hasattr(data['gt_map_seg_mask'], 'data') else data['gt_map_seg_mask']
            if isinstance(gt_map_seg, torch.Tensor):
                gt_map_seg = gt_map_seg.cpu().numpy()
            if len(gt_map_seg.shape) == 3:
                map_viz = gt_map_seg[0] if gt_map_seg.shape[0] == 1 else gt_map_seg.max(axis=0)
            else:
                map_viz = gt_map_seg
            axes[0].imshow(map_viz, cmap='tab20')
            axes[0].set_title('Map Segmentation')
        else:
            axes[0].text(0.5, 0.5, 'No Map Data', ha='center', va='center')
            axes[0].set_title('Map Segmentation (N/A)')
        axes[0].axis('off')
        
        # Object seg
        if has_obj:
            gt_obj_seg = data['gt_object_seg_mask'].data if hasattr(data['gt_object_seg_mask'], 'data') else data['gt_object_seg_mask']
            if isinstance(gt_obj_seg, torch.Tensor):
                gt_obj_seg = gt_obj_seg.cpu().numpy()
            if len(gt_obj_seg.shape) == 3:
                obj_viz = gt_obj_seg[0] if gt_obj_seg.shape[0] == 1 else gt_obj_seg.max(axis=0)
            else:
                obj_viz = gt_obj_seg
            axes[1].imshow(obj_viz, cmap='tab20', vmin=0, vmax=10)
            axes[1].set_title('Object Segmentation')
        else:
            axes[1].text(0.5, 0.5, 'No Object Data', ha='center', va='center')
            axes[1].set_title('Object Segmentation (N/A)')
        axes[1].axis('off')
        
        # Combined overlay
        if has_map and has_obj:
            # Create RGB overlay
            combined = np.zeros((*map_viz.shape, 3), dtype=np.uint8)
            
            # Map in blue channel
            combined[:, :, 2] = (map_viz > 0).astype(np.uint8) * 255
            
            # Objects in red/green channels
            combined[:, :, 0] = (obj_viz > 0).astype(np.uint8) * 255
            combined[:, :, 1] = (obj_viz > 0).astype(np.uint8) * 128
            
            axes[2].imshow(combined)
            axes[2].set_title('Combined (Blue=Map, Red=Objects)')
        else:
            axes[2].text(0.5, 0.5, 'Need Both Map & Object Data', ha='center', va='center')
            axes[2].set_title('Combined (N/A)')
        axes[2].axis('off')
        
        plt.suptitle(f'Sample {idx} - Ground Truth Visualization', fontsize=16)
        save_path = output_dir / f'sample_{idx:04d}_combined.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved combined: {save_path}")
        plt.close()
    
    print(f"{'='*60}\n")


def main():
    # Load config
    config_path = 'projects/configs/bevformer/bevformer_small_seg_maptr.py'
    
    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Build dataset
    print("\nBuilding dataset...")
    dataset = build_dataset(cfg.data.train)
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")
    
    # Output directory
    output_dir = Path('./vis_gt_data')
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving visualizations to: {output_dir}\n")
    
    # Visualize first N samples
    num_samples = 5
    print(f"Visualizing {num_samples} samples...\n")
    
    for idx in range(min(num_samples, len(dataset))):
        try:
            visualize_sample(dataset, idx, output_dir)
        except Exception as e:
            print(f"\n✗ Error visualizing sample {idx}:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print(f"\n{'='*60}")
    print(f"✓ Done! Check visualizations in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()