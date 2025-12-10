import copy
import torch
import torch.nn as nn
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16


@DETECTORS.register_module()
class BEVFormerMapTR(MVXTwoStageDetector):
    """
    BEVFormer + MapTR with Segmentation-based Masking.
    
    This detector combines:
    1. BEVFormer encoder for temporal BEV feature extraction
    2. Segmentation head for semantic understanding
    3. Segmentation-based masking of BEV features
    4. MapTR decoder for HD map prediction
    
    Args:
        video_test_mode (bool): Use temporal information during inference
        use_grid_mask (bool): Apply grid mask augmentation
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 freeze_seg=False):  # NEW: Freeze segmentation flag

        super(BEVFormerMapTR, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)
        
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.freeze_seg = freeze_seg
        
        # Temporal information for video test mode
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_pts_train(self,
                        pts_feats,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        img_metas,
                        gt_object_seg_mask=None, 
                        gt_map_seg_mask=None,
                        gt_bboxes_ignore=None,
                        prev_bev=None):
        """Forward function for point cloud branch during training."""
        # Forward through head
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)

        # Compute losses
        losses = dict()
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses_pts = self.pts_bbox_head.loss(
            *loss_inputs, 
            img_metas=img_metas,
            object_seg_gt=gt_object_seg_mask,
            map_seg_gt=gt_map_seg_mask,
        )
        losses.update(losses_pts)
        
        return losses, outs

    def forward_dummy(self, img):
        """Dummy forward for ONNX export."""
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """
        Obtain history BEV features iteratively.
        
        To save GPU memory, gradients are not calculated.
        This method processes the temporal queue to build up BEV features.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                
                # Check if previous BEV exists
                if not img_metas[0].get('prev_bev_exists', True):
                    prev_bev = None
                
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                
                # Only compute BEV features (encoder only)
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                    points=None,
                    img_metas=None,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    gt_object_seg_mask=None,
                    gt_map_seg_mask=None,
                    gt_pv_seg_mask=None,
                    gt_labels=None,
                    gt_bboxes=None,
                    img=None,
                    proposals=None,
                    gt_bboxes_ignore=None,
                    img_depth=None,
                    img_mask=None):
        """Forward training function."""
        # ============================================================
        # PART 1: Handle temporal queue (BEVFormer)
        # ============================================================
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]  # Current frame
        
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None
        
        # Get current frame metas
        img_metas = [each[len_queue-1] for each in img_metas]
        
        # Check if previous BEV exists
        if not img_metas[0].get('prev_bev_exists', True):
            prev_bev = None
        
        # ============================================================
        # PART 2: Extract image features
        # ============================================================
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # ============================================================
        # PART 3: Forward and compute losses
        # ============================================================
        losses = dict()
        losses_pts, outs = self.forward_pts_train(
            img_feats, 
            gt_bboxes_3d, 
            gt_labels_3d,
            img_metas, 
            gt_object_seg_mask=gt_object_seg_mask, 
            gt_map_seg_mask=gt_map_seg_mask,
            gt_bboxes_ignore=gt_bboxes_ignore, 
            prev_bev=prev_bev
        )
        
        losses.update(losses_pts)

        # ============================================================
        # Cache outputs for visualization
        # ============================================================
        if hasattr(self, 'pts_bbox_head'):
            # Store in model for hook to access
            self._cached_outputs = {
                'gt_object_seg_mask': gt_object_seg_mask,
                'pred_seg': outs.get('seg'),
                'bev_embed': outs.get('bev_embed'),
                # Note: masked_bev_embed would need to be returned from head
                'pred_pts': outs.get('all_pts_preds'),
                'pred_labels': outs.get('all_cls_scores'),
                'pred_scores': torch.sigmoid(outs['all_cls_scores']) if outs.get('all_cls_scores') is not None else None,
                'gt_pts': gt_bboxes_3d,
                'gt_labels': gt_labels_3d,
            }

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        """
        Forward test function.
        
        Handles temporal information for video test mode.
        """
        if isinstance(img_metas[0], list):
            # Nested list format: [[{meta}]]
            img_metas_inner = img_metas[0]
        else:
            # Single list format: [{meta}]
            img_metas_inner = img_metas
        
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        
        img = [img] if img is None else img

        # Scene change detection
        if img_metas_inner[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # First sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        
        # Update scene token
        self.prev_frame_info['scene_token'] = img_metas_inner[0]['scene_token']

        # Do not use temporal information if video_test_mode is False
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get ego motion delta
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        # Simple test
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        
        # Save BEV features and ego motion
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """
        Convert detection results to a list of numpy arrays.
        
        Args:
            bboxes: Bounding boxes
            scores: Prediction scores
            labels: Class labels
            pts: Points for map vectors
            attrs: Additional attributes
            
        Returns:
            result_dict: Dictionary of results
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """
        Test function for point cloud branch.
        
        Returns:
            bev_embed: BEV features
            seg_preds: Segmentation predictions
            bbox_results: Map prediction results
        """
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]
        
        seg_preds = outs.get('seg', None)
        
        return outs['bev_embed'], seg_preds, bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """
        Test function without augmentation.
        
        Returns:
            new_prev_bev: Updated BEV features for next frame
            result_list: List of results (bbox + segmentation)
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        result_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, seg_preds, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        
        for result_dict, pts_bbox in zip(result_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['seg_preds'] = seg_preds

        return new_prev_bev, result_list