"""
Visualize with GT - Creates custom pipeline that includes GT data
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from copy import deepcopy

import projects.mmdet3d_plugin


def create_viz_pipeline(cfg):
    """Create a visualization pipeline that includes GT data."""
    
    # Start with test pipeline but enable GT
    viz_pipeline = [
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
        dict(type='NormalizeMultiviewImage', **cfg.img_norm_cfg),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),  # ← LOAD GT!
        dict(type='RasterizeMapVectors', map_grid_conf=cfg.map_grid_conf),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    with_gt=True,  # ← ENABLE GT!
                    with_label=True,  # ← ENABLE LABELS!
                    class_names=cfg.map_classes),
                dict(type='CustomCollect3D', keys=['img', 'semantic_indices', 'gt_bboxes_3d', 'gt_labels_3d'])  # ← COLLECT GT!
            ])
    ]
    
    return viz_pipeline


def extract_from_container(data):
    """Recursively extract data from DataContainer."""
    if hasattr(data, 'data'):
        return extract_from_container(data.data)
    elif isinstance(data, list) and len(data) > 0:
        return extract_from_container(data[0])
    else:
        return data


def draw_map_vectors(ax, vectors, labels, scores=None, pc_range=[-30, -15, 30, 15], 
                     classes=['divider', 'ped_crossing', 'boundary'], 
                     colors=None, alpha=1.0, linewidth=2, title='', is_gt=True):
    """Draw vectorized map on axis."""
    if colors is None:
        colors = {
            0: (1.0, 0.0, 0.0),   # divider - red
            1: (0.0, 1.0, 0.0),   # ped_crossing - green
            2: (0.0, 0.5, 1.0),   # boundary - blue
        }
    
    ax.set_xlim(pc_range[0], pc_range[2])
    ax.set_ylim(pc_range[1], pc_range[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    
    # Ego vehicle
    ego_size = 2.0
    ego_rect = Rectangle((-ego_size/2, -ego_size/2), ego_size, ego_size,
                        linewidth=2, edgecolor='black', facecolor='yellow', alpha=0.5, zorder=10)
    ax.add_patch(ego_rect)
    
    if vectors is None or len(vectors) == 0:
        ax.text(0, 0, 'No vectors', ha='center', va='center', fontsize=14, color='red')
        ax.set_title(title, fontsize=12, fontweight='bold')
        return
    
    # Adaptive threshold
    threshold = 0.3
    if not is_gt and scores is not None and len(scores) > 0:
        max_score = scores.max()
        if max_score < 0.3:
            threshold = max(0.05, max_score * 0.5)
    
    num_drawn = 0
    for i, (vec, label) in enumerate(zip(vectors, labels)):
        if not is_gt and scores is not None and scores[i] < threshold:
            continue
        
        color = colors.get(int(label), (0.5, 0.5, 0.5))
        
        if vec.shape[0] > 1:
            ax.plot(vec[:, 0], vec[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=5)
            num_drawn += 1
            
            if not is_gt and scores is not None and num_drawn <= 10:
                mid_idx = len(vec) // 2
                ax.text(vec[mid_idx, 0], vec[mid_idx, 1], f'{scores[i]:.2f}', 
                       fontsize=7, bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))
    
    ax.set_title(f'{title} ({num_drawn} vectors)', fontsize=12, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], linewidth=2, label=classes[i])
                      for i in range(len(classes)) if i in colors]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def visualize_sample(model, data, idx, save_dir, pc_range, map_classes):
    """Visualize a single sample."""
    
    print(f"Sample {idx}...", end=' ')
    
    try:
        # Extract GT Object Segmentation
        gt_obj_seg = None
        if 'gt_object_seg_mask' in data:
            gt_data = extract_from_container(data['gt_object_seg_mask'])
            if isinstance(gt_data, torch.Tensor):
                gt_obj_seg = gt_data.cpu().numpy()
            elif isinstance(gt_data, np.ndarray):
                gt_obj_seg = gt_data
            if gt_obj_seg is not None and gt_obj_seg.ndim > 2:
                while gt_obj_seg.ndim > 2 and gt_obj_seg.shape[0] == 1:
                    gt_obj_seg = gt_obj_seg[0]
        
        # Extract GT Map Vectors
        gt_map_vectors = []
        gt_map_labels = []
        
        if 'gt_bboxes_3d' in data:
            gt_bboxes = extract_from_container(data['gt_bboxes_3d'])
            
            if gt_bboxes is not None:
                # Check type
                if hasattr(gt_bboxes, 'instance_list'):
                    # LiDARInstanceLines format
                    for instance in gt_bboxes.instance_list:
                        coords = np.array(list(instance.coords))
                        if coords.shape[0] > 1:
                            gt_map_vectors.append(coords)
                elif hasattr(gt_bboxes, 'tensor'):
                    # Tensor format - might need different handling
                    print(f"  [GT in tensor format: {gt_bboxes.tensor.shape}]")
            
            if 'gt_labels_3d' in data:
                gt_labels_raw = extract_from_container(data['gt_labels_3d'])
                if isinstance(gt_labels_raw, torch.Tensor):
                    gt_map_labels = gt_labels_raw.cpu().numpy()
                elif isinstance(gt_labels_raw, np.ndarray):
                    gt_map_labels = gt_labels_raw
                else:
                    gt_map_labels = np.array(gt_labels_raw) if gt_labels_raw else []
        
        # Forward Pass
        in_data = {i: j for i, j in data.items() if 'img' in i}
        
        model.eval()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **in_data)
            
            if not result or len(result) == 0:
                print("✗ Empty result")
                return False
            
            # Extract Predictions
            pred_obj_seg = None
            pred_map_vectors = []
            pred_map_labels = []
            pred_map_scores = []
            
            if 'seg_preds' in result[0] and result[0]['seg_preds'] is not None:
                pred_seg = result[0]['seg_preds']
                pred_obj_seg = pred_seg.argmax(dim=1)[0].cpu().numpy()
            
            if 'pts_bbox' in result[0] and result[0]['pts_bbox'] is not None:
                pts_bbox = result[0]['pts_bbox']
                if 'pts_3d' in pts_bbox:
                    pred_pts = pts_bbox['pts_3d'].cpu().numpy()
                    pred_map_labels = pts_bbox['labels_3d'].cpu().numpy()
                    pred_map_scores = pts_bbox['scores_3d'].cpu().numpy()
                    for i in range(len(pred_pts)):
                        pred_map_vectors.append(pred_pts[i])
        
        # Create Visualization
        has_seg = (gt_obj_seg is not None or pred_obj_seg is not None)
        has_map = (len(gt_map_vectors) > 0 or len(pred_map_vectors) > 0)
        
        if not has_seg and not has_map:
            print("✗ No data")
            return False
        
        # Setup figure
        if has_seg and has_map:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax_seg_gt = fig.add_subplot(gs[0, 0])
            ax_seg_pred = fig.add_subplot(gs[0, 1])
            ax_seg_err = fig.add_subplot(gs[0, 2])
            ax_map_gt = fig.add_subplot(gs[1, 0])
            ax_map_pred = fig.add_subplot(gs[1, 1])
            ax_map_overlay = fig.add_subplot(gs[1, 2])
        elif has_seg:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax_seg_gt, ax_seg_pred, ax_seg_err = axes
            ax_map_gt = ax_map_pred = ax_map_overlay = None
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax_map_gt, ax_map_pred, ax_map_overlay = axes
            ax_seg_gt = ax_seg_pred = ax_seg_err = None
        
        # Plot Segmentation
        accuracy = 0.0
        if has_seg and gt_obj_seg is not None and pred_obj_seg is not None:
            num_classes = max(gt_obj_seg.max(), pred_obj_seg.max()) + 1
            
            im1 = ax_seg_gt.imshow(gt_obj_seg, cmap='tab20', vmin=0, vmax=num_classes-1, 
                                  origin='lower', interpolation='nearest')
            ax_seg_gt.set_title('GT Object Segmentation', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=ax_seg_gt, fraction=0.046, pad=0.04)
            
            im2 = ax_seg_pred.imshow(pred_obj_seg, cmap='tab20', vmin=0, vmax=num_classes-1,
                                    origin='lower', interpolation='nearest')
            ax_seg_pred.set_title('Predicted Object Segmentation', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=ax_seg_pred, fraction=0.046, pad=0.04)
            
            diff = (gt_obj_seg != pred_obj_seg).astype(float)
            im3 = ax_seg_err.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1,
                                   origin='lower', interpolation='nearest')
            accuracy = 100 * (1 - diff.mean())
            ax_seg_err.set_title(f'Error Map\nAccuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')
            plt.colorbar(im3, ax=ax_seg_err, fraction=0.046, pad=0.04)
        
        # Plot HD Map
        pc_range_plot = [pc_range[0], pc_range[1], pc_range[3], pc_range[4]]
        
        if has_map:
            if ax_map_gt is not None:
                draw_map_vectors(ax_map_gt, gt_map_vectors if len(gt_map_vectors) > 0 else None,
                               gt_map_labels if len(gt_map_labels) > 0 else [],
                               pc_range=pc_range_plot, classes=map_classes,
                               linewidth=2.5, title='GT HD Map Vectors', is_gt=True)
            
            if ax_map_pred is not None:
                draw_map_vectors(ax_map_pred, pred_map_vectors if len(pred_map_vectors) > 0 else None,
                               pred_map_labels if len(pred_map_labels) > 0 else [],
                               scores=pred_map_scores if len(pred_map_scores) > 0 else None,
                               pc_range=pc_range_plot, classes=map_classes,
                               linewidth=2.5, title='Predicted HD Map Vectors', is_gt=False)
            
            if ax_map_overlay is not None:
                ax_map_overlay.set_xlim(pc_range_plot[0], pc_range_plot[2])
                ax_map_overlay.set_ylim(pc_range_plot[1], pc_range_plot[3])
                ax_map_overlay.set_aspect('equal')
                ax_map_overlay.grid(True, alpha=0.3)
                
                ego_rect = Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor='black', 
                                   facecolor='yellow', alpha=0.5, zorder=10)
                ax_map_overlay.add_patch(ego_rect)
                
                colors = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.5, 1.0)}
                
                # GT solid
                for vec, label in zip(gt_map_vectors, gt_map_labels):
                    if vec.shape[0] > 1:
                        color = colors.get(int(label), (0.5, 0.5, 0.5))
                        ax_map_overlay.plot(vec[:, 0], vec[:, 1], color=color, 
                                          linewidth=3, linestyle='-', alpha=0.6, zorder=5)
                
                # Pred dashed
                threshold = 0.3
                if len(pred_map_scores) > 0 and pred_map_scores.max() < 0.3:
                    threshold = max(0.05, pred_map_scores.max() * 0.5)
                
                for vec, label, score in zip(pred_map_vectors, pred_map_labels, pred_map_scores):
                    if score >= threshold and vec.shape[0] > 1:
                        color = colors.get(int(label), (0.5, 0.5, 0.5))
                        ax_map_overlay.plot(vec[:, 0], vec[:, 1], color=color,
                                          linewidth=2, linestyle='--', alpha=0.8, zorder=6)
                
                ax_map_overlay.set_title('Overlay: GT (solid) + Pred (dashed)', fontsize=12, fontweight='bold')
                ax_map_overlay.set_xlabel('X (m)')
                ax_map_overlay.set_ylabel('Y (m)')
        
        # Title and Save
        title_parts = []
        if pred_obj_seg is not None:
            title_parts.append(f'Object Seg ({pred_obj_seg.max() + 1} cls)')
        if has_map:
            title_parts.append(f'HD Map ({len(pred_map_vectors)} pred, {len(gt_map_vectors)} GT)')
        
        fig.suptitle(f'Sample {idx} - {" | ".join(title_parts)}', fontsize=16, fontweight='bold')
        
        save_path = os.path.join(save_dir, f'sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        result_str = f"✓ seg_acc={accuracy:.1f}% map:{len(gt_map_vectors)}GT/{len(pred_map_vectors)}pred"
        print(result_str)
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--out-dir', default='work_dirs/visualizations_with_gt')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION WITH CUSTOM GT PIPELINE")
    print(f"{'='*80}\n")
    
    cfg = Config.fromfile(args.config)
    
    # Create dataset with CUSTOM visualization pipeline
    print("Creating custom visualization pipeline...")
    viz_pipeline = create_viz_pipeline(cfg)
    
    dataset_cfg = deepcopy(cfg.data.val)
    dataset_cfg.pipeline = viz_pipeline  # ← Use custom pipeline!
    
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    
    dataset = build_dataset(dataset_cfg)
    print(f"✓ Dataset: {len(dataset)} samples with GT data\n")
    
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Visualizing...\n")
    
    success = 0
    for idx, data in enumerate(data_loader):
        if idx >= args.num_samples:
            break
        if visualize_sample(model, data, idx, args.out_dir, cfg.point_cloud_range, cfg.map_classes):
            success += 1
    
    print(f"\n{'='*80}")
    print(f"✅ Success: {success}/{args.num_samples}")
    print(f"✅ Output: {args.out_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()