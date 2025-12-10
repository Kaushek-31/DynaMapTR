import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmcv.runner import BaseModule
from mmcv.runner import force_fp32

@HEADS.register_module()
class MapTRDecoder(BaseModule):
    """MapTR head for HD map prediction from masked BEV features"""
    
    def __init__(self,
                 num_queries=50,
                 in_channels=256,
                 num_classes=3,
                 num_points=20,
                 coord_dim=2,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_reg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        
        self.num_queries = num_queries
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_points = num_points
        self.coord_dim = coord_dim
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, in_channels)
        
        # Transformer decoder (build from config)
        from mmdet.models import build_transformer
        self.transformer_decoder = build_transformer(transformer_decoder)
        
        # Output heads
        self.cls_head = nn.Linear(in_channels, num_classes)
        self.reg_head = nn.Linear(in_channels, num_points * coord_dim)
        
        # Losses
        from mmdet.models.builder import build_loss
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
    
    def forward(self, bev_features, img_metas):
        """
        Args:
            bev_features: [B, C, H, W] - Masked BEV features
            img_metas: Image meta information
        Returns:
            dict: Predictions including class scores and coordinates
        """
        bs = bev_features.shape[0]
        
        # Prepare query embeddings
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # Flatten BEV features for attention
        bev_feat_flat = bev_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Decoder forward
        # This is simplified - you'll need to adapt to your transformer decoder API
        hs = self.transformer_decoder(
            query=query_embeds,
            key=bev_feat_flat,
            value=bev_feat_flat,
        )
        
        # Output heads
        cls_scores = self.cls_head(hs)  # [B, num_queries, num_classes]
        reg_coords = self.reg_head(hs)  # [B, num_queries, num_points*2]
        
        # Reshape coordinates
        reg_coords = reg_coords.reshape(bs, self.num_queries, self.num_points, self.coord_dim)
        
        return {
            'cls_scores': cls_scores,
            'reg_coords': reg_coords
        }
    
    @force_fp32(apply_to=('preds_dict',))
    def loss(self, preds_dict, gt_vectors):
        """
        Compute losses
        Args:
            preds_dict: Predictions from forward()
            gt_vectors: Ground truth vectorized map
        """
        cls_scores = preds_dict['cls_scores']
        reg_coords = preds_dict['reg_coords']
        
        # You'll need to implement matching and loss computation
        # This is a placeholder
        loss_cls = self.loss_cls(cls_scores, gt_labels)
        loss_reg = self.loss_reg(reg_coords, gt_coords)
        
        return {
            'loss_cls': loss_cls,
            'loss_reg': loss_reg
        }