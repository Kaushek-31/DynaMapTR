import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from ..modules.builder import build_seg_encoder
from mmseg.models.builder import build_loss as build_seg_loss
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2


def compute_seg_metrics(seg_pred, seg_gt, num_classes=11):
    """Compute per-class IoU for better segmentation monitoring."""
    pred_labels = seg_pred.argmax(dim=1)  # (B, H, W)
    
    metrics = {}  # Dictionary, not list!
    for cls in range(num_classes):
        pred_mask = (pred_labels == cls)
        gt_mask = (seg_gt == cls)
        
        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
            metrics[f'seg_iou_cls{cls}'] = iou.item()
    
    # Also compute mean IoU (excluding background)
    fg_ious = [v for k, v in metrics.items() if 'cls0' not in k and v > 0]
    if fg_ious:
        metrics['seg_mIoU_fg'] = sum(fg_ious) / len(fg_ious)
    
    return metrics  # Returns dict, not list

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """Calculate BEV parameters for grid transformation."""
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)
    return bev_resolution, bev_start_position, bev_dimension


class BevFeatureSlicer(nn.Module):
    """Crop interested area in BEV feature for semantic map segmentation."""
    
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()

        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])
            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # Convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])
            
            self.map_grid = torch.stack(torch.meshgrid(
                self.norm_map_x, self.norm_map_y), dim=2).permute(1, 0, 2)

    def forward(self, x):
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)
            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return new_pts


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return bboxes


@HEADS.register_module()
class BEVFormerMapTRHead(DETRHead):
    """
    BEVFormer + MapTR Head with Segmentation-based Masking.
    
    Flow:
    1. BEVFormer encoder generates BEV features
    2. Segmentation head predicts semantic classes
    3. Mask BEV features based on segmentation output
    4. MapTR decoder predicts HD map elements from masked features
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_vec_one2one=50,
                 num_vec_one2many=0,
                 k_one2many=0,
                 lambda_one2many=1,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 aux_seg=dict(
                    use_aux_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=4,  # Changed default to 4
                    feat_down_sample=32,
                ),
                seg_mask_classes=[],
                freeze_seg=False,
                freeze_encoder_and_above=False,
                seg_gt_size=(200, 400),  # NEW: Explicit GT size
                det_grid_conf=None,
                map_grid_conf=None,
                 z_cfg=dict(
                     pred_z_flag=False,
                     gt_z_flag=False,
                 ),
                 loss_pts=dict(type='ChamferDistance',
                               loss_src_weight=1.0,
                               loss_dst_weight=1.0),
                 loss_seg=dict(type='SimpleLoss',
                               pos_weight=2.13,
                               loss_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        
        if 'code_size' in kwargs:
            self.code_size = 2 if not z_cfg['pred_z_flag'] else 3
        else:
            self.code_size = 2
        
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern

        num_vec = num_vec_one2one + num_vec_one2many
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.aux_seg = aux_seg
        self.z_cfg = z_cfg
        
        # NEW: Segmentation masking
        self.seg_mask_classes = seg_mask_classes
        self.freeze_seg = freeze_seg
        self.freeze_encoder_and_above = freeze_encoder_and_above
        self.seg_gt_size = seg_gt_size  # NEW
        self.det_grid_conf = det_grid_conf
        self.map_grid_conf = map_grid_conf

        super(BEVFormerMapTRHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)

        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

        if isinstance(loss_seg, list):
            # Explicit multi-loss handling
            self.loss_seg = nn.ModuleList(
                [build_loss(ls) for ls in loss_seg]
            )
        else:
            self.loss_seg = build_loss(loss_seg)

        self.curr_epoch = 0
        self.debug_epoch_int = self.train_cfg.get('debug_iter', 100)
        self.save_dir = self.train_cfg.get('save_dir', "work_dirs/debug_visualizations")
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch, regression branch, and segmentation head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        # Segmentation head
        if self.aux_seg['use_aux_seg'] and self.aux_seg['bev_seg']:
            # BEV feature cropper
            self.feat_cropper = BevFeatureSlicer(self.det_grid_conf, self.map_grid_conf)
            
            # Segmentation decoder - USE SegEncode
            from ..modules.builder import build_seg_encoder
            # self.seg_head = build_seg_encoder(dict(
            #     type='SegEncode',
            #     inC=self.embed_dims,  # 256
            #     outC=self.aux_seg['seg_classes'],  # 4
            #     size=self.seg_gt_size  # (200, 400)
            # ))
            self.seg_head = build_seg_encoder(dict(
                type='SegEncodeASPP',  # or 'SegEncodeV2'
                inC=self.embed_dims,  # 256
                outC=self.aux_seg['seg_classes'],  # 3
                size=self.seg_gt_size  # (200, 400)
            ))

        # Query embeddings
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the head."""
        self.transformer.init_weights()

        if hasattr(self, 'seg_pretrained') and self.seg_pretrained:
            self._load_seg_pretrained_weights()

        # Initialize classification branches
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        
        # Freeze segmentation head if requested
        if self.freeze_seg and self.aux_seg['use_aux_seg']:
            self._freeze_segmentation_head()
        
        if self.freeze_encoder_and_above:
            self._freeze_encoder_and_above()
    
    def _load_seg_pretrained_weights(self):
        """Load pretrained weights for segmentation head."""
        import torch
        from mmcv.runner import load_checkpoint
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f'Loading segmentation pretrained weights from {self.seg_pretrained}')
        
        try:
            checkpoint = load_checkpoint(
                self, 
                self.seg_pretrained, 
                map_location='cpu',
                strict=False,
                logger=logger
            )
            logger.info('Successfully loaded segmentation pretrained weights')
        except Exception as e:
            logger.warning(f'Failed to load segmentation pretrained weights: {e}')
            logger.warning('Training will continue with random initialization')
    
    def _freeze_encoder_and_above(self):
        """
        Freeze everything EXCEPT the MapTR Decoder.
        Frozen: Backbone (handled by config), Neck (handled by config), Encoder, Segmentation Head.
        Trainable: MapTR Decoder, Classification/Regression Branches.
        """
        print(f"\n{'='*40}")
        print(f"❄️  FREEZING ENCODER & SEGMENTATION HEAD ❄️")
        
        # 1. Freeze Segmentation Head (if it exists)
        if hasattr(self, 'seg_head'):
            self.seg_head.eval()
            for param in self.seg_head.parameters():
                param.requires_grad = False
            print(" - SegHead: Frozen")

        # 2. Freeze Transformer Encoder (BEVFormer Encoder)
        if hasattr(self.transformer, 'encoder'):
            self.transformer.encoder.eval()
            for param in self.transformer.encoder.parameters():
                param.requires_grad = False
            print(" - Encoder: Frozen")
            
        # 3. Freeze Embeddings (Positional, Level, Cams)
        # These are often overlooked but are part of the 'input' stage
        if hasattr(self.transformer, 'level_embeds'):
            self.transformer.level_embeds.requires_grad = False
        if hasattr(self.transformer, 'cams_embeds'):
            self.transformer.cams_embeds.requires_grad = False
        print(" - Embeddings: Frozen")

        print(f"{'='*40}\n")

    def _freeze_segmentation_head(self):
        """Freeze segmentation-related parameters ONLY. Encoder stays trainable."""
        import logging
        logger = logging.getLogger(__name__)
        
        frozen_params = 0
        trainable_params = 0
        frozen_modules = []
        
        # ============================================================
        # FREEZE: Segmentation head only
        # ============================================================
        if self.aux_seg['bev_seg'] and hasattr(self, 'seg_head'):
            for name, param in self.seg_head.named_parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_modules.append('seg_head')
            
            # Set to eval mode permanently
            self.seg_head.eval()
            for module in self.seg_head.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)):
                    module.eval()
        
        # Freeze feature cropper if it has parameters
        if hasattr(self, 'feat_cropper'):
            for name, param in self.feat_cropper.named_parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_modules.append('feat_cropper')
        
        # ============================================================
        # VERIFY: Encoder and decoder are TRAINABLE
        # ============================================================
        encoder_trainable = 0
        decoder_trainable = 0
        other_trainable = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                if 'encoder' in name.lower():
                    encoder_trainable += param.numel()
                elif 'decoder' in name.lower() or 'cls_branches' in name or 'reg_branches' in name:
                    decoder_trainable += param.numel()
                else:
                    other_trainable += param.numel()
        
        # ============================================================
        # LOGGING
        # ============================================================
        logger.info(f'{"="*60}')
        logger.info(f'STAGE 2 PARAMETER FREEZE STATUS')
        logger.info(f'{"="*60}')
        logger.info(f'FROZEN modules: {frozen_modules}')
        logger.info(f'FROZEN params:  {frozen_params:,}')
        logger.info(f'{"="*60}')
        logger.info(f'TRAINABLE params: {trainable_params:,}')
        logger.info(f'  - Encoder:      {encoder_trainable:,}')
        logger.info(f'  - Decoder:      {decoder_trainable:,}')
        logger.info(f'  - Other:        {other_trainable:,}')
        logger.info(f'{"="*60}')
        
        # Sanity check: encoder should be trainable
        if encoder_trainable == 0:
            logger.warning('⚠️  WARNING: Encoder has 0 trainable params!')
            logger.warning('    This may cause poor MapTR predictions.')
        
        if decoder_trainable == 0:
            logger.warning('⚠️  WARNING: Decoder has 0 trainable params!')
        
        print(f"[Stage 2] Frozen: {frozen_params:,} params | Trainable: {trainable_params:,} params")

    def train(self, mode=True):
        """Override train to keep segmentation in eval mode if frozen."""
        super().train(mode)
        
        # Keep frozen segmentation head in eval mode
        if self.freeze_seg and hasattr(self, 'seg_head'):
            self.seg_head.eval()
            # Force eval mode for batch norm and dropout
            for module in self.seg_head.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)):
                    module.eval()
        if self.freeze_encoder_and_above:
            if hasattr(self.transformer, 'encoder'):
                self.transformer.encoder.eval()
            # Force eval mode for batch norm and dropout in encoder
            for module in self.transformer.encoder.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)):
                    module.eval()

    def apply_segmentation_mask(self, bev_features, seg_pred):
        """
        Apply segmentation mask to BEV features.
        """
        # Get segmentation predictions
        seg_classes = torch.argmax(seg_pred, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Create mask: 1 for keep, 0 for mask out
        mask = torch.ones_like(seg_classes, dtype=torch.float32)
        for mask_class in self.seg_mask_classes:
            mask = mask * (seg_classes != mask_class).float()
            
        # === CHANGE: SOFT MASKING ===
        # Instead of multiplying by 0 (Hard Mask), multiply by a small value (0.1).
        # This allows gradients to flow back to the encoder even through "masked" regions.
        # It tells the model: "Focus here (1.0), but don't be blind elsewhere (0.1)"
        
        soft_mask = mask * 0.9 + 0.1  # Foreground=1.0, Background=0.1
        
        # Apply mask to BEV features
        masked_bev_features = bev_features * soft_mask
        
        return masked_bev_features

    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """
        Forward function with proper encoder → segmentation → masking → decoder flow.
        
        Flow:
        1. BEVFormer encoder generates BEV features
        2. Generate segmentation predictions from BEV features
        3. Apply segmentation mask to BEV features
        4. MapTR decoder predicts HD map from MASKED features
        """
        if self.training:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one
        
        
        if hasattr(self, 'seg_head') and not hasattr(self, '_checked_weights'):
            try:
                # Get the first parameter of the head, whatever it is named
                first_param = next(self.seg_head.parameters())
                
                w_sum = first_param.sum()
                is_frozen = not first_param.requires_grad
                
                print(f"\n{'='*40}")
                print(f"DEBUG: Segmentation Head Check")
                print(f" - Weight Sum: {w_sum:.4f}")
                print(f" - Frozen Status: {'FROZEN (Correct)' if is_frozen else 'TRAINABLE (Warning!)'}")
                print(f"{'='*40}\n")
                
                self._checked_weights = True
            except StopIteration:
                print("WARNING: seg_head has no parameters to check!")
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        # Prepare query embeddings
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                            device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # Attention mask for one2one and one2many
        self_attn_mask = (
            torch.zeros([num_vec, num_vec,]).bool().to(mlvl_feats[0].device)
        )
        self_attn_mask[self.num_vec_one2one:, 0:self.num_vec_one2one,] = True
        self_attn_mask[0:self.num_vec_one2one, self.num_vec_one2one:,] = True

        if only_bev:
            # Only return BEV features (for temporal queue building)
            output_dict = self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat=None,
                bev_queries=bev_queries,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            return output_dict['bev']
        
        # ============================================================
        # STEP 1: Get BEV features from ENCODER ONLY
        # ============================================================
        output_dict = self.transformer.get_bev_features(
            mlvl_feats,
            lidar_feat=None,
            bev_queries=bev_queries,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )
        
        if only_bev:
            return output_dict  # ← Return dict: {'bev': tensor, 'depth': tensor}

        bev_embed = output_dict['bev']  # (B, H*W, C) or (H*W, B, C)
        depth = output_dict['depth']
        
        # print("BEV Embed Shape:", bev_embed.shape)

        # ============================================================
        # STEP 2: Generate Segmentation Predictions
        # ============================================================
        outputs_seg = None
        masked_bev_embed = bev_embed  # Default: no masking
        
        if self.aux_seg['use_aux_seg'] and self.aux_seg['bev_seg']:
            # Reshape BEV features for segmentation: (H*W, B, C) → (B, C, H, W)
            if bev_embed.shape[0] == self.bev_h * self.bev_w:
                # Format: (H*W, B, C)
                seg_bev_embed = bev_embed.permute(1, 0, 2)  # (B, H*W, C)
            else:
                # Format: (B, H*W, C)
                seg_bev_embed = bev_embed
            
            seg_bev_embed = seg_bev_embed.view(bs, self.bev_h, self.bev_w, -1)  # (B, H, W, C)
            seg_bev_embed = seg_bev_embed.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            
            # NO ROTATION - work entirely in LiDAR coordinate system
            # Crop to appropriate region (if grids differ)
            # seg_bev_embed = self.feat_cropper(seg_bev_embed)
            
            # Predict segmentation
            outputs_seg = self.seg_head(seg_bev_embed)  # (B, num_classes, H_seg, W_seg)
            
            # ============================================================
            # STEP 3: Apply Segmentation Mask
            # ============================================================
            if len(self.seg_mask_classes) > 0:
                # Resize segmentation to match BEV feature resolution
                if outputs_seg.shape[-2:] != (self.bev_h, self.bev_w):
                    seg_pred_resized = F.interpolate(
                        outputs_seg,
                        size=(self.bev_h, self.bev_w),
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    seg_pred_resized = outputs_seg
                
                # Apply mask to BEV features
                # Reconstruct (B, C, H, W) for masking
                if bev_embed.shape[0] == self.bev_h * self.bev_w:
                    bev_for_mask = bev_embed.permute(1, 0, 2).view(bs, self.bev_h, self.bev_w, -1)
                else:
                    bev_for_mask = bev_embed.view(bs, self.bev_h, self.bev_w, -1)
                
                bev_for_mask = bev_for_mask.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
                
                # Apply masking
                masked_bev = self.apply_segmentation_mask(bev_for_mask, seg_pred_resized)
                # masked_bev = bev_for_mask
                
                # Reshape back to (H*W, B, C) for decoder
                masked_bev_embed = masked_bev.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
            else:
                # No masking - keep original format
                if bev_embed.shape[0] != self.bev_h * self.bev_w:
                    # Convert (B, H*W, C) → (H*W, B, C)
                    masked_bev_embed = bev_embed.permute(1, 0, 2)
                else:
                    masked_bev_embed = bev_embed
        else:
            # No segmentation - ensure proper format for decoder
            if bev_embed.shape[0] != self.bev_h * self.bev_w:
                masked_bev_embed = bev_embed.permute(1, 0, 2)
            else:
                masked_bev_embed = bev_embed
        
        # ============================================================
        # STEP 4: Decoder on MASKED BEV Features
        # ============================================================
        hs, init_reference, inter_references = self.transformer.forward_decoder(
            bev_embed=masked_bev_embed,  # ← MASKED features!
            mlvl_feats=mlvl_feats,
            object_query_embed=object_query_embeds,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            self_attn_mask=self_attn_mask,
            num_vec=num_vec,
            num_pts_per_vec=self.num_pts_per_vec,
        )
        
        # ============================================================
        # Process Decoder Outputs
        # ============================================================
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference[..., 0:2]
            else:
                reference = inter_references[lvl - 1][..., 0:2]
            
            reference = inverse_sigmoid(reference)
            
            outputs_class = self.cls_branches[lvl](
                hs[lvl].view(bs, num_vec, self.num_pts_per_vec, -1).mean(2)
            )
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp = tmp[..., 0:2]
            tmp += reference
            tmp = tmp.sigmoid()
            
            outputs_coord, outputs_pts_coord = self.transform_box(tmp, num_vec=num_vec)

            outputs_classes_one2one.append(outputs_class[:, 0:self.num_vec_one2one])
            outputs_coords_one2one.append(outputs_coord[:, 0:self.num_vec_one2one])
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, 0:self.num_vec_one2one])

            outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:])
            outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:])
            outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        outs = {
            'bev_embed': masked_bev_embed,
            'bev_embed_plain': bev_embed,
            'mlvl_feats': mlvl_feats,
            'all_cls_scores': outputs_classes_one2one,
            'all_bbox_preds': outputs_coords_one2one,
            'all_pts_preds': outputs_pts_coords_one2one,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None,
            'seg': outputs_seg,  # Segmentation at original resolution
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_cls_scores=None,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                seg=None,
            )
        }

        return outs

    def transform_box(self, pts, num_vec=50, y_first=False):
        """Convert points set into bounding box."""
        pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        
        if self.transform_method == 'minmax':
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        
        return bbox, pts_reshape

    def _get_target_single(self, cls_score, bbox_pred, pts_pred, gt_labels,
                           gt_bboxes, gt_shifts_pts, gt_bboxes_ignore=None):
        """Compute regression and classification targets for one image."""
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]
        
        assign_result, order_index = self.assigner.assign(
            bbox_pred, cls_score, pts_pred,
            gt_bboxes, gt_labels, gt_shifts_pts,
            gt_bboxes_ignore
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # Label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # Bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # Pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        
        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights, pos_inds, neg_inds)

    def get_targets(self, cls_scores_list, bbox_preds_list, pts_preds_list,
                    gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """Compute regression and classification targets for a batch."""
        assert gt_bboxes_ignore_list is None
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self, cls_scores, bbox_preds, pts_preds, gt_bboxes_list,
                    gt_labels_list, gt_shifts_pts_list, gt_bboxes_ignore_list=None):
        """Loss function for outputs from a single decoder layer."""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
            gt_bboxes_ignore_list
        )
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # Classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # Bbox loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], avg_factor=num_total_pos
        )

        # Pts loss
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec),
                                      mode='linear', align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :], normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :], avg_factor=num_total_pos
        )
        
        # Direction loss
        dir_weights = pts_weights[:, :-self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval:, :] - denormed_pts_preds[:, :-self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval:, :] - pts_targets[:, :-self.dir_interval, :]
        
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :], pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :], avg_factor=num_total_pos
        )

        # IoU loss
        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], avg_factor=num_total_pos
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts,
            object_seg_gt=None, map_seg_gt=None,
            gt_bboxes_ignore=None, img_metas=None):
        
        """Loss function with both object and map segmentation."""
        assert gt_bboxes_ignore is None
        self.curr_epoch  += 1  # For visualization purposes
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds = preds_dicts['all_pts_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_pts_preds = preds_dicts['enc_pts_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        
        if self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list
            ]
        else:
            raise NotImplementedError

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list
        )

        # -------------------------------
        #  SEGMENTATION GT & PRED (COLORIZED + METRICS)
        # -------------------------------
        if 'seg' in preds_dicts and preds_dicts['seg'] is not None:
            if self.curr_epoch % self.debug_epoch_int == 0:
                b = 0
                # ----- 1. Extract prediction -----
                seg_pred_raw = preds_dicts['seg'][b]                      # [C, H, W]
                seg_pred = seg_pred_raw.argmax(0).detach().cpu().numpy()  # [H, W]

                # ----- 2. Extract GT -----
                if object_seg_gt is not None:
                    seg_gt = object_seg_gt[b]
                elif map_seg_gt is not None:
                    seg_gt = map_seg_gt[b]
                else:
                    seg_gt = None

                # Convert GT to numpy H×W
                if isinstance(seg_gt, torch.Tensor):
                    seg_gt = seg_gt.squeeze()
                    seg_gt = seg_gt.cpu().numpy()

                # Safety check
                if seg_gt is None:
                    print("⚠ Debug: No seg_gt available at this batch. Skipping seg visualization.")
                else:

                    # ----- 3. Define color palette -----
                    class_colors = {
                        0: (0, 0, 0),        # background: black
                        1: (255, 50, 50),    # vehicles: red
                        2: (30, 144, 255)    # pedestrian: blue
                    }
                    class_names = ["Background", "Vehicle", "Pedestrian"]

                    def colorize(mask):
                        h, w = mask.shape
                        out = np.zeros((h, w, 3), dtype=np.uint8)
                        for cid, col in class_colors.items():
                            out[mask == cid] = col
                        return out

                    pred_rgb = colorize(seg_pred)
                    gt_rgb = colorize(seg_gt)

                    # ----- 4. Compute IoU metrics safely -----
                    with torch.no_grad():
                        gt_tensor = torch.from_numpy(seg_gt).long().to(seg_pred_raw.device)
                        metrics = compute_seg_metrics(seg_pred_raw.unsqueeze(0),  # ensure shape [B,C,H,W]
                                                    gt_tensor.unsqueeze(0),
                                                    num_classes=3)

                    metric_text = "\n".join([
                        f"IoU(bg)  = {metrics.get('seg_iou_cls0', 0):.3f}",
                        f"IoU(veh) = {metrics.get('seg_iou_cls1', 0):.3f}",
                        f"IoU(ped) = {metrics.get('seg_iou_cls2', 0):.3f}",
                        f"mIoU_fg  = {metrics.get('seg_mIoU_fg', 0):.3f}"
                    ])

                    # ----- 5. Plot everything -----
                    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

                    axs[0].imshow(gt_rgb)
                    axs[0].set_title("GT Segmentation", fontsize=14)
                    axs[0].axis("off")

                    axs[1].imshow(pred_rgb)
                    axs[1].set_title("Pred Segmentation", fontsize=14)
                    axs[1].axis("off")

                    # ----- 6. Legend -----
                    import matplotlib.patches as mpatches
                    patches = []
                    for cid, cname in enumerate(class_names):
                        patches.append(
                            mpatches.Patch(color=np.array(class_colors[cid]) / 255.0, label=cname)
                        )
                    fig.legend(handles=patches, loc="upper right", fontsize=12)

                    # ----- 7. Add IoU metrics box -----
                    fig.text(0.02, 0.02, metric_text, fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.7))

                    plt.tight_layout()
                    fig.savefig(f"{self.save_dir}/seg_epoch{self.curr_epoch}.png")
                    plt.close(fig)
        # -------------------------------
        #  MAPTR DECODER OUTPUT VISUALIZATION
        # -------------------------------
        if self.curr_epoch % self.debug_epoch_int == 0:
            try:
                b = 0  # Batch index
                
                # ----- 1. Extract predictions (last decoder layer) -----
                cls_scores = all_cls_scores[-1][b]  # [num_vec, num_classes]
                pts_preds = all_pts_preds[-1][b]    # [num_vec, num_pts, 2]
                
                # Get predicted classes and confidence
                pred_scores, pred_labels = cls_scores.sigmoid().max(dim=-1)  # [num_vec]
                
                # Denormalize predicted points to real coordinates
                pts_preds_denorm = denormalize_2d_pts(pts_preds, self.pc_range)  # [num_vec, num_pts, 2]
                
                # ----- 2. Extract GT -----
                gt_labels = gt_labels_list[b].cpu().numpy()  # [num_gt]
                gt_pts = gt_pts_list[b].cpu().numpy()        # [num_gt, num_pts, 2]
                
                # ----- 3. Define colors for map classes -----
                map_class_colors = {
                    0: (255, 0, 0),      # divider: red
                    1: (0, 255, 0),      # ped_crossing: green  
                    2: (0, 0, 255),      # boundary: blue
                }
                map_class_names = ["Divider", "Ped Crossing", "Boundary"]
                
                # ----- 4. Create BEV plot -----
                fig, axs = plt.subplots(1, 2, figsize=(16, 8))
                
                # Set BEV plot limits (based on pc_range)
                x_min, y_min = self.pc_range[0], self.pc_range[1]
                x_max, y_max = self.pc_range[3], self.pc_range[4]
                
                for ax in axs:
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                
                # ----- 5. Plot GT polylines -----
                axs[0].set_title(f"GT Map Elements (N={len(gt_labels)})", fontsize=14)
                for i, (label, pts) in enumerate(zip(gt_labels, gt_pts)):
                    color = np.array(map_class_colors.get(label, (128, 128, 128))) / 255.0
                    axs[0].plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2, alpha=0.8)
                    axs[0].scatter(pts[:, 0], pts[:, 1], c=[color], s=10, alpha=0.6)
                
                # ----- 6. Plot predicted polylines -----
                # Filter by confidence threshold
                conf_threshold = 0.3
                pred_scores_np = pred_scores.detach().cpu().numpy()
                pred_labels_np = pred_labels.detach().cpu().numpy()
                pts_preds_np = pts_preds_denorm.detach().cpu().numpy()
                
                valid_mask = pred_scores_np > conf_threshold
                num_valid = valid_mask.sum()
                
                axs[1].set_title(f"Pred Map Elements (N={num_valid}, thresh={conf_threshold})", fontsize=14)
                
                for i in range(len(pred_scores_np)):
                    if pred_scores_np[i] > conf_threshold:
                        label = pred_labels_np[i]
                        pts = pts_preds_np[i]
                        score = pred_scores_np[i]
                        
                        color = np.array(map_class_colors.get(label, (128, 128, 128))) / 255.0
                        axs[1].plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2, alpha=score)
                        axs[1].scatter(pts[:, 0], pts[:, 1], c=[color], s=10, alpha=score * 0.6)
                
                # ----- 7. Add legend -----
                import matplotlib.patches as mpatches
                patches = []
                for cid, cname in enumerate(map_class_names):
                    patches.append(
                        mpatches.Patch(color=np.array(map_class_colors[cid]) / 255.0, label=cname)
                    )
                fig.legend(handles=patches, loc="upper right", fontsize=12)
                
                # ----- 8. Add metrics text -----
                # Count predictions per class
                pred_counts = {c: 0 for c in range(len(map_class_names))}
                gt_counts = {c: 0 for c in range(len(map_class_names))}
                
                for label in pred_labels_np[valid_mask]:
                    if label in pred_counts:
                        pred_counts[label] += 1
                for label in gt_labels:
                    if label in gt_counts:
                        gt_counts[label] += 1
                
                metric_text = "Class Counts (GT / Pred):\n"
                for cid, cname in enumerate(map_class_names):
                    metric_text += f"  {cname}: {gt_counts.get(cid, 0)} / {pred_counts.get(cid, 0)}\n"
                
                # Add loss values
                metric_text += f"\nloss_cls: {losses_cls[-1].item():.4f}\n"
                metric_text += f"loss_pts: {losses_pts[-1].item():.4f}\n"
                metric_text += f"loss_dir: {losses_dir[-1].item():.4f}"
                
                fig.text(0.02, 0.02, metric_text, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8), family='monospace')
                
                plt.tight_layout()
                fig.savefig(f"{self.save_dir}/maptr_epoch{self.curr_epoch}.png", dpi=150)
                plt.close(fig)
                
            except Exception as e:
                print(f"⚠ MapTR visualization failed: {e}")
        # -------------------------------
        #  MASKED vs PLAIN BEV FEATURE COMPARISON
        # -------------------------------
        if self.curr_epoch % self.debug_epoch_int == 0 and len(self.seg_mask_classes) > 0:
            try:
                b = 0  # Batch index
                
                # ----- 1. Get BEV features -----
                masked_bev = preds_dicts['bev_embed']  # (H*W, B, C) or similar
                plain_bev = preds_dicts.get('bev_embed_plain', None)
                
                if plain_bev is not None:
                    # Reshape to (B, C, H, W) for visualization
                    if masked_bev.shape[0] == self.bev_h * self.bev_w:
                        # Format: (H*W, B, C)
                        masked_bev_vis = masked_bev.permute(1, 0, 2)[b]  # (H*W, C)
                        plain_bev_vis = plain_bev.permute(1, 0, 2)[b] if plain_bev.shape[0] == self.bev_h * self.bev_w else plain_bev[b]
                    else:
                        # Format: (B, H*W, C)
                        masked_bev_vis = masked_bev[b]  # (H*W, C)
                        plain_bev_vis = plain_bev[b]
                    
                    # Reshape to (H, W, C)
                    masked_bev_vis = masked_bev_vis.view(self.bev_h, self.bev_w, -1).detach().cpu().numpy()
                    plain_bev_vis = plain_bev_vis.view(self.bev_h, self.bev_w, -1).detach().cpu().numpy()
                    
                    # ----- 2. Compute feature magnitude (L2 norm across channels) -----
                    masked_magnitude = np.linalg.norm(masked_bev_vis, axis=-1)
                    plain_magnitude = np.linalg.norm(plain_bev_vis, axis=-1)
                    
                    # Compute difference
                    diff_magnitude = plain_magnitude - masked_magnitude
                    
                    # ----- 3. Get segmentation mask for overlay -----
                    seg_pred = None
                    if 'seg' in preds_dicts and preds_dicts['seg'] is not None:
                        seg_pred = preds_dicts['seg'][b].argmax(0).detach().cpu().numpy()
                        # Resize if needed
                        if seg_pred.shape != (self.bev_h, self.bev_w):
                            seg_pred = cv2.resize(seg_pred.astype(np.float32), 
                                                  (self.bev_w, self.bev_h), 
                                                  interpolation=cv2.INTER_NEAREST).astype(np.int32)
                    
                    # ----- 4. Create visualization -----
                    fig, axs = plt.subplots(2, 3, figsize=(36, 12))
                    
                    # Row 1: BEV Feature Magnitudes
                    im0 = axs[0, 0].imshow(plain_magnitude, cmap='viridis', aspect='auto')
                    axs[0, 0].set_title("Plain BEV Feature Magnitude", fontsize=12)
                    axs[0, 0].axis('off')
                    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046)
                    
                    im1 = axs[0, 1].imshow(masked_magnitude, cmap='viridis', aspect='auto')
                    axs[0, 1].set_title("Masked BEV Feature Magnitude", fontsize=12)
                    axs[0, 1].axis('off')
                    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046)
                    
                    im2 = axs[0, 2].imshow(diff_magnitude, cmap='RdBu_r', aspect='auto',
                                           vmin=-np.abs(diff_magnitude).max(), 
                                           vmax=np.abs(diff_magnitude).max())
                    axs[0, 2].set_title("Difference (Plain - Masked)", fontsize=12)
                    axs[0, 2].axis('off')
                    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046)
                    
                    # Row 2: Segmentation mask overlay and masked regions
                    if seg_pred is not None:
                        # Create mask visualization
                        mask_vis = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)
                        mask_vis[seg_pred == 0] = [50, 50, 50]    # background: dark gray
                        mask_vis[seg_pred == 1] = [255, 50, 50]   # vehicles: red
                        mask_vis[seg_pred == 2] = [50, 50, 255]   # pedestrian: blue
                        
                        axs[1, 0].imshow(mask_vis, aspect='auto')
                        axs[1, 0].set_title("Segmentation Prediction", fontsize=12)
                        axs[1, 0].axis('off')
                        
                        # Show which regions are masked
                        masked_regions = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)
                        for mask_cls in self.seg_mask_classes:
                            masked_regions[seg_pred == mask_cls] = 1.0
                        
                        axs[1, 1].imshow(masked_regions, cmap='Reds', aspect='auto', vmin=0, vmax=1)
                        axs[1, 1].set_title(f"Masked Regions (classes {self.seg_mask_classes})", fontsize=12)
                        axs[1, 1].axis('off')
                        
                        # Overlay masked regions on feature magnitude
                        overlay = plain_magnitude.copy()
                        overlay_rgb = plt.cm.viridis(overlay / (overlay.max() + 1e-8))[:, :, :3]
                        overlay_rgb[masked_regions > 0] = [1, 0, 0]  # Red for masked regions
                        
                        axs[1, 2].imshow(overlay_rgb, aspect='auto')
                        axs[1, 2].set_title("Plain BEV with Masked Regions (Red)", fontsize=12)
                        axs[1, 2].axis('off')
                    else:
                        for ax in axs[1, :]:
                            ax.axis('off')
                    
                    # Add stats text
                    stats_text = (
                        f"Plain BEV: mean={plain_magnitude.mean():.2f}, max={plain_magnitude.max():.2f}\n"
                        f"Masked BEV: mean={masked_magnitude.mean():.2f}, max={masked_magnitude.max():.2f}\n"
                        f"Masked pixels: {(masked_magnitude == 0).sum()} / {masked_magnitude.size}\n"
                        f"Mask classes: {self.seg_mask_classes}"
                    )
                    fig.text(0.02, 0.02, stats_text, fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.8), family='monospace')
                    
                    plt.tight_layout()
                    fig.savefig(f"{self.save_dir}/bev_mask_comparison_epoch{self.curr_epoch}.png", dpi=150)
                    plt.close(fig)
                    
            except Exception as e:
                print(f"⚠ BEV mask comparison visualization failed: {e}")
                import traceback
                traceback.print_exc()

        # -------------------------------
        #  MAPTR: MASKED vs PLAIN BEV PREDICTIONS COMPARISON
        # -------------------------------
        if self.curr_epoch % self.debug_epoch_int == 0 and len(self.seg_mask_classes) > 0:
            try:
                b = 0
                plain_bev = preds_dicts.get('bev_embed_plain', None)
                
                if plain_bev is not None:
                    # Run decoder with PLAIN (unmasked) BEV features for comparison
                    with torch.no_grad():
                        # Prepare plain BEV for decoder
                        if plain_bev.shape[0] != self.bev_h * self.bev_w:
                            plain_bev_decoder = plain_bev.permute(1, 0, 2)
                        else:
                            plain_bev_decoder = plain_bev
                        
                        # Get query embeddings
                        num_vec = self.num_vec_one2one
                        if self.query_embed_type == 'instance_pts':
                            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
                            instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)
                            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(plain_bev.dtype)
                        else:
                            object_query_embeds = self.query_embedding.weight.to(plain_bev.dtype)
                        
                        # Self attention mask
                        self_attn_mask = torch.zeros([num_vec, num_vec]).bool().to(plain_bev.device)
                        mlvl_feats_stored = preds_dicts.get('mlvl_feats', None)
                        if mlvl_feats_stored is None:
                            raise ValueError("mlvl_feats not available for comparison")
                        
                        # Run decoder with plain BEV
                        hs_plain, init_ref_plain, inter_refs_plain = self.transformer.forward_decoder(
                            bev_embed=plain_bev_decoder,
                            mlvl_feats=mlvl_feats_stored,  # Not needed for decoder
                            object_query_embed=object_query_embeds,
                            bev_h=self.bev_h,
                            bev_w=self.bev_w,
                            reg_branches=self.reg_branches if self.with_box_refine else None,
                            cls_branches=self.cls_branches if self.as_two_stage else None,
                            self_attn_mask=self_attn_mask,
                            num_vec=num_vec,
                            num_pts_per_vec=self.num_pts_per_vec,
                        )
                        
                        # Process plain BEV decoder outputs (last layer only)
                        hs_plain = hs_plain.permute(0, 2, 1, 3)
                        lvl = -1  # Last layer
                        reference = inter_refs_plain[lvl - 1][..., 0:2] if lvl > 0 else init_ref_plain[..., 0:2]
                        reference = inverse_sigmoid(reference)
                        
                        bs = hs_plain.shape[1]
                        cls_plain = self.cls_branches[-1](
                            hs_plain[-1].view(bs, num_vec, self.num_pts_per_vec, -1).mean(2)
                        )
                        pts_plain = self.reg_branches[-1](hs_plain[-1])
                        pts_plain = pts_plain[..., 0:2] + reference
                        pts_plain = pts_plain.sigmoid()
                        _, pts_plain = self.transform_box(pts_plain, num_vec=num_vec)
                    
                    # ----- Get masked predictions (already computed) -----
                    cls_masked = all_cls_scores[-1][b]
                    pts_masked = all_pts_preds[-1][b]
                    
                    # Get plain predictions
                    cls_plain = cls_plain[b]
                    pts_plain = pts_plain[b]
                    
                    # Convert to numpy
                    pred_scores_masked, pred_labels_masked = cls_masked.sigmoid().max(dim=-1)
                    pred_scores_plain, pred_labels_plain = cls_plain.sigmoid().max(dim=-1)
                    
                    pts_masked_np = denormalize_2d_pts(pts_masked, self.pc_range).detach().cpu().numpy()
                    pts_plain_np = denormalize_2d_pts(pts_plain, self.pc_range).detach().cpu().numpy()
                    
                    scores_masked_np = pred_scores_masked.detach().cpu().numpy()
                    scores_plain_np = pred_scores_plain.detach().cpu().numpy()
                    labels_masked_np = pred_labels_masked.detach().cpu().numpy()
                    labels_plain_np = pred_labels_plain.detach().cpu().numpy()
                    
                    # GT
                    gt_labels_np = gt_labels_list[b].cpu().numpy()
                    gt_pts_np = gt_pts_list[b].cpu().numpy()
                    
                    # ----- Create comparison plot -----
                    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
                    
                    map_class_colors = {
                        0: (255, 0, 0),      # divider: red
                        1: (0, 255, 0),      # ped_crossing: green  
                        2: (0, 0, 255),      # boundary: blue
                    }
                    map_class_names = ["Divider", "Ped Crossing", "Boundary"]
                    
                    x_min, y_min = self.pc_range[0], self.pc_range[1]
                    x_max, y_max = self.pc_range[3], self.pc_range[4]
                    conf_threshold = 0.3
                    
                    for ax in axs:
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel('X (m)')
                        ax.set_ylabel('Y (m)')
                    
                    # Plot GT
                    axs[0].set_title(f"GT Map Elements (N={len(gt_labels_np)})", fontsize=14)
                    for label, pts in zip(gt_labels_np, gt_pts_np):
                        color = np.array(map_class_colors.get(label, (128, 128, 128))) / 255.0
                        axs[0].plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2, alpha=0.8)
                        axs[0].scatter(pts[:, 0], pts[:, 1], c=[color], s=10, alpha=0.6)
                    
                    # Plot Plain BEV predictions
                    valid_plain = scores_plain_np > conf_threshold
                    axs[1].set_title(f"Plain BEV Predictions (N={valid_plain.sum()})", fontsize=14)
                    for i in range(len(scores_plain_np)):
                        if scores_plain_np[i] > conf_threshold:
                            color = np.array(map_class_colors.get(labels_plain_np[i], (128, 128, 128))) / 255.0
                            axs[1].plot(pts_plain_np[i, :, 0], pts_plain_np[i, :, 1], '-', 
                                       color=color, linewidth=2, alpha=scores_plain_np[i])
                            axs[1].scatter(pts_plain_np[i, :, 0], pts_plain_np[i, :, 1], 
                                          c=[color], s=10, alpha=scores_plain_np[i] * 0.6)
                    
                    # Plot Masked BEV predictions
                    valid_masked = scores_masked_np > conf_threshold
                    axs[2].set_title(f"Masked BEV Predictions (N={valid_masked.sum()})", fontsize=14)
                    for i in range(len(scores_masked_np)):
                        if scores_masked_np[i] > conf_threshold:
                            color = np.array(map_class_colors.get(labels_masked_np[i], (128, 128, 128))) / 255.0
                            axs[2].plot(pts_masked_np[i, :, 0], pts_masked_np[i, :, 1], '-', 
                                       color=color, linewidth=2, alpha=scores_masked_np[i])
                            axs[2].scatter(pts_masked_np[i, :, 0], pts_masked_np[i, :, 1], 
                                          c=[color], s=10, alpha=scores_masked_np[i] * 0.6)
                    
                    # Legend
                    import matplotlib.patches as mpatches
                    patches = [mpatches.Patch(color=np.array(map_class_colors[cid]) / 255.0, label=cname) 
                               for cid, cname in enumerate(map_class_names)]
                    fig.legend(handles=patches, loc="upper right", fontsize=12)
                    
                    # Stats
                    stats_text = (
                        f"Confidence threshold: {conf_threshold}\n"
                        f"Plain predictions: {valid_plain.sum()}\n"
                        f"Masked predictions: {valid_masked.sum()}\n"
                        f"GT elements: {len(gt_labels_np)}\n"
                        f"Mask classes: {self.seg_mask_classes}"
                    )
                    fig.text(0.02, 0.02, stats_text, fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.8), family='monospace')
                    
                    plt.tight_layout()
                    fig.savefig(f"{self.save_dir}/maptr_masked_vs_plain_epoch{self.curr_epoch}.png", dpi=150)
                    plt.close(fig)
                    
            except Exception as e:
                print(f"⚠ Masked vs Plain MapTR comparison failed: {e}")
                import traceback
                traceback.print_exc()

        
        loss_dict = dict()
        
        # Segmentation loss
        if self.aux_seg['use_aux_seg'] and self.aux_seg['bev_seg']:
            if 'seg' in preds_dicts and preds_dicts['seg'] is not None:
                seg_output = preds_dicts['seg']
                num_imgs = seg_output.size(0)
                
                # Get GT segmentation
                if object_seg_gt is not None:
                    seg_gt = object_seg_gt
                elif map_seg_gt is not None:
                    seg_gt = torch.stack([map_seg_gt[i] for i in range(num_imgs)], dim=0)
                else:
                    seg_gt = None
                
                # Ensure seg_gt is always a tensor of shape [B, H, W] or [B, 1, H, W]
                if isinstance(seg_gt, list):
                    seg_gt = torch.stack([x for x in seg_gt], dim=0)

                if seg_gt is not None:
                    # Squeeze if needed: [B, 1, H, W] → [B, H, W]
                    if seg_gt.dim() == 4 and seg_gt.shape[1] == 1:
                        seg_gt = seg_gt.squeeze(1)
                    
                    # Resize if shape mismatch
                    if seg_output.shape[2:] != seg_gt.shape[1:]:
                        seg_gt = F.interpolate(
                            seg_gt.unsqueeze(1).float(),
                            size=seg_output.shape[2:],
                            mode='nearest'
                        ).squeeze(1)
                    
                    loss_seg = self.loss_seg(seg_output, seg_gt.long())
                    loss_dict['loss_seg'] = loss_seg
                    
                    with torch.no_grad():
                        seg_metrics = compute_seg_metrics(seg_output, seg_gt, num_classes=self.aux_seg['seg_classes'])
                        for k, v in seg_metrics.items():
                            loss_dict[k] = torch.tensor(v, device=seg_output.device)

                    # Log accuracy
                    # with torch.no_grad():
                    #     seg_pred = seg_output.argmax(dim=1)
                    #     seg_acc = (seg_pred == seg_gt).float().mean()
                    #     loss_dict['seg_acc'] = seg_acc

        # Encoder losses
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_pts, enc_losses_dir = \
                self.loss_single(enc_cls_scores, enc_bbox_preds, enc_pts_preds,
                                 gt_bboxes_list, binary_labels_list, gt_pts_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_losses_iou'] = enc_losses_iou
            loss_dict['enc_losses_pts'] = enc_losses_pts
            loss_dict['enc_losses_dir'] = enc_losses_dir

        # Last decoder layer losses
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        
        # Other decoder layer losses
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(
                losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1],
                losses_pts[:-1], losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions."""
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']
            ret_list.append([bboxes, scores, labels, pts])

        return ret_list