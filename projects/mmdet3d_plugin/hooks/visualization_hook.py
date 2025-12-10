"""
Custom hook for visualizing predictions during training.
Saves map predictions, segmentation masks, and BEV features every N iterations.
"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mmcv.runner import HOOKS, Hook
from shapely.geometry import LineString
from PIL import Image


@HOOKS.register_module()
class VisualizationHook(Hook):
    """
    Visualize training progress every N iterations.
    
    Saves:
    1. Map predictions (vectorized)
    2. Object segmentation (GT + Pred)
    3. Map segmentation (GT + Pred)
    4. BEV features (before/after masking)
    """
    
    def __init__(self,
                 interval=50,
                 save_dir='work_dirs/visualizations',
                 pc_range=[-30.0, -15.0, -10.0, 30.0, 15.0, 10.0],
                 map_classes=['divider', 'ped_crossing', 'boundary'],
                 object_classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']):
        self.interval = interval
        self.save_dir = save_dir
        self.pc_range = pc_range
        self.map_classes = map_classes
        self.object_classes = object_classes
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Color maps
        self.object_colors = self._get_object_colors()
        self.map_colors = self._get_map_colors()
    
    def _get_object_colors(self):
        """Get colors for object classes."""
        return {
            0: (255, 0, 0),      # car - red
            1: (0, 255, 0),      # truck - green
            2: (0, 0, 255),      # construction - blue
            3: (255, 255, 0),    # bus - yellow
            4: (255, 0, 255),    # trailer - magenta
            5: (0, 255, 255),    # barrier - cyan
            6: (128, 0, 0),      # motorcycle - dark red
            7: (0, 128, 0),      # bicycle - dark green
            8: (0, 0, 128),      # pedestrian - dark blue
            9: (128, 128, 0),    # traffic_cone - olive
        }
    
    def _get_map_colors(self):
        """Get colors for map classes."""
        return {
            0: (255, 100, 100),  # divider - light red
            1: (100, 255, 100),  # ped_crossing - light green
            2: (100, 100, 255),  # boundary - light blue
        }
    
    def after_train_iter(self, runner):
        """Called after each training iteration."""
        if not self.every_n_iters(runner, self.interval):
            return
        
        # Get current iteration
        iter_num = runner.iter
        
        print(f"\n{'='*60}")
        print(f"Saving visualizations at iteration {iter_num}")
        print(f"{'='*60}")
        
        # Create iteration directory
        iter_dir = os.path.join(self.save_dir, f'iter_{iter_num:06d}')
        os.makedirs(iter_dir, exist_ok=True)
        
        try:
            # Get model outputs from last forward pass
            # This requires modifying the detector to cache outputs
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            
            if not hasattr(model, '_cached_outputs'):
                print("Warning: Model doesn't have cached outputs. Skipping visualization.")
                return
            
            cached = model._cached_outputs
            
            # 1. Visualize segmentation
            self._visualize_segmentation(
                cached.get('gt_object_seg_mask'),
                cached.get('pred_seg'),
                iter_dir
            )
            
            # 2. Visualize BEV features
            self._visualize_bev_features(
                cached.get('bev_embed'),
                cached.get('masked_bev_embed'),
                iter_dir
            )
            
            # 3. Visualize map predictions
            self._visualize_map_predictions(
                cached.get('pred_pts'),
                cached.get('pred_labels'),
                cached.get('pred_scores'),
                cached.get('gt_pts'),
                cached.get('gt_labels'),
                iter_dir
            )
            
            print(f"✓ Visualizations saved to {iter_dir}")
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_segmentation(self, gt_mask, pred_logits, save_dir):
        """Visualize segmentation masks."""
        if gt_mask is None or pred_logits is None:
            return
        
        # Get first sample in batch
        gt = gt_mask[0].cpu().numpy()  # (1, H, W) or (H, W)
        pred = pred_logits[0].cpu().numpy()  # (C, H, W)
        
        # Remove channel dim if present
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]
        
        # Get predicted classes
        pred_classes = pred.argmax(axis=0)  # (H, W)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # GT
        axes[0].imshow(gt, cmap='tab10', vmin=0, vmax=10)
        axes[0].set_title('GT Object Segmentation')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(pred_classes, cmap='tab10', vmin=0, vmax=10)
        axes[1].set_title('Predicted Object Segmentation')
        axes[1].axis('off')
        
        # Difference
        diff = (gt != pred_classes).astype(np.float32)
        axes[2].imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[2].set_title('Segmentation Error')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'segmentation.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_bev_features(self, bev_embed, masked_bev_embed, save_dir):
        """Visualize BEV features before and after masking."""
        if bev_embed is None:
            return
        
        # Get first sample: (H*W, B, C) or (B, H*W, C)
        if bev_embed.shape[1] > bev_embed.shape[0]:
            # (H*W, B, C) → (B, H*W, C)
            bev = bev_embed.permute(1, 0, 2)[0]  # (H*W, C)
        else:
            bev = bev_embed[0]  # (H*W, C)
        
        # Reshape to 2D: assume square-ish
        total = bev.shape[0]
        h = int(np.sqrt(total))
        w = total // h
        
        # Take mean across channels
        bev_2d = bev.view(h, w, -1).mean(dim=-1).cpu().numpy()
        
        fig, axes = plt.subplots(1, 2 if masked_bev_embed is None else 3, figsize=(15, 5))
        
        # Original BEV
        axes[0].imshow(bev_2d, cmap='viridis')
        axes[0].set_title('BEV Features (Original)')
        axes[0].axis('off')
        
        # Masked BEV (if available)
        if masked_bev_embed is not None:
            if masked_bev_embed.shape[1] > masked_bev_embed.shape[0]:
                masked = masked_bev_embed.permute(1, 0, 2)[0]
            else:
                masked = masked_bev_embed[0]
            
            masked_2d = masked.view(h, w, -1).mean(dim=-1).cpu().numpy()
            
            axes[1].imshow(masked_2d, cmap='viridis')
            axes[1].set_title('BEV Features (After Masking)')
            axes[1].axis('off')
            
            # Mask effect
            mask_effect = np.abs(bev_2d - masked_2d)
            axes[2].imshow(mask_effect, cmap='hot')
            axes[2].set_title('Masking Effect')
            axes[2].axis('off')
        else:
            # Just show feature histogram
            axes[1].hist(bev_2d.flatten(), bins=50)
            axes[1].set_title('Feature Distribution')
            axes[1].set_xlabel('Feature Value')
            axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bev_features.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_map_predictions(self, pred_pts, pred_labels, pred_scores, 
                                   gt_pts, gt_labels, save_dir):
        """Visualize map vector predictions."""
        if pred_pts is None:
            return
        
        # Get first sample in batch
        pred_pts = pred_pts[-1][0].cpu().numpy()  # Last decoder layer, first sample
        pred_labels = pred_labels[-1][0].cpu().numpy()
        pred_scores = pred_scores[-1][0].cpu().numpy()
        
        # Create canvas
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Set limits based on pc_range
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Plot predictions
        axes[0].set_title('Predicted Map Vectors')
        for i, (pts, label, score) in enumerate(zip(pred_pts, pred_labels, pred_scores)):
            if score > 0.3:  # Confidence threshold
                pts_denorm = self._denormalize_pts(pts)  # (num_pts, 2)
                color = np.array(self.map_colors.get(label, (128, 128, 128))) / 255.0
                axes[0].plot(pts_denorm[:, 0], pts_denorm[:, 1], 
                           color=color, linewidth=2, alpha=score)
        
        # Plot GT (if available)
        if gt_pts is not None and gt_labels is not None:
            axes[1].set_title('Ground Truth Map Vectors')
            
            # GT format depends on your data structure
            # Assuming gt_pts is LiDARInstanceLines
            if hasattr(gt_pts[0], 'instance_list'):
                gt_instances = gt_pts[0].instance_list
                gt_labels_list = gt_labels[0].cpu().numpy()
                
                for instance, label in zip(gt_instances, gt_labels_list):
                    coords = np.array(list(instance.coords))
                    color = np.array(self.map_colors.get(label, (128, 128, 128))) / 255.0
                    axes[1].plot(coords[:, 0], coords[:, 1], 
                               color=color, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'map_vectors.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _denormalize_pts(self, pts):
        """Denormalize predicted points from [0, 1] to pc_range."""
        pts_denorm = pts.copy()
        
        # X dimension
        pts_denorm[:, 0] = pts[:, 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        
        # Y dimension
        pts_denorm[:, 1] = pts[:, 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        
        return pts_denorm