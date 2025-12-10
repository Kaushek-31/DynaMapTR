# _base_ = ['./bevformer_small_seg_maptr.py']

# # Stage 2: Freeze segmentation, train MapTR decoder

# model = dict(
#     pts_bbox_head=dict(
#         # Segmentation config
#         aux_seg=dict(
#             use_aux_seg=True,
#             bev_seg=True,
#             pv_seg=False,
#             seg_classes=11,
#             feat_down_sample=32,
#         ),
#         seg_mask_classes=[1, 9],  # NOW enable masking (car, pedestrian)
#         freeze_seg=True,  # ← FREEZE segmentation head
        
#         # ENABLE MapTR losses
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=2.0),  # ← Enabled
#         loss_bbox=dict(type='L1Loss', loss_weight=0.0),  # Keep 0 for MapTR
#         loss_iou=dict(type='GIoULoss', loss_weight=0.0),  # Keep 0 for MapTR
#         loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),  # ← Enabled
#         loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),  # ← Enabled
        
#         # Keep seg loss for monitoring (optional, can set to 0)
#         loss_seg=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=False,
#             loss_weight=0.0,  # ← Set to 0 since frozen (or keep small for monitoring)
#             class_weight=[0.5] + [2.0]*10),
#     ),
#     train_cfg=dict(pts=dict(
#         debug_vis=True,
#     ))
# )

# # Training settings for MapTR
# total_epochs = 24
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     min_lr_ratio=1e-3)

# # Load from Stage 1 checkpoint
# load_from = 'work_dirs/stage1_seg_only/epoch_24.pth'  # ← Your best seg checkpoint
# resume_from = None

# checkpoint_config = dict(max_keep_ckpts=3, interval=2)
# work_dir = 'work_dirs/stage2_maptr_frozen_seg'



_base_ = ['./bevformer_small_seg_maptr.py']

# Stage 2: Freeze segmentation, train MapTR decoder

model = dict(
    pts_bbox_head=dict(
        # Segmentation config - must match Stage 1
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=False,
            seg_classes=3,
            feat_down_sample=8,
        ),
        
        # Must match Stage 1 architecture
        seg_head_type='SegEncodeASPP',  # ← Match Stage 1
        
        # Enable masking with trained segmentation
        seg_mask_classes=[1, 2],  # ← vehicle=1, pedestrian=2 (check your class indices)
        mask_threshold=0.5,  # Confidence threshold for masking
        
        freeze_seg=True,  # ← FREEZE segmentation head
        freeze_encoder_and_above=True,  # ← Freeze encoder and above to retain seg features

        # ENABLE MapTR losses
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=3.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.05),
        loss_iou=dict(type='GIoULoss', loss_weight=0.5),
        loss_pts=dict(type='PtsL1Loss', loss_weight=4.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.5),
        
        # Seg loss for monitoring only (frozen, so gradients won't flow)
        loss_seg=dict(
            type='FocalDiceLossV2',  # ← Match Stage 1
            focal_weight=1.0,
            dice_weight=1.0,
            gamma=2.0,
            alpha=0.25,
            class_weights=[0.05, 3.0, 10.0],  # ← Match Stage 1
            loss_weight= 0.0,  # ← 0 since frozen (set to 0.01 if you want to monitor)
        ),
    ),
    train_cfg=dict(pts=dict(
        debug_vis=True,
        debug_iter=50,
        save_dir = 'work_dirs/stage4_maptr_frozen_seg_mini/debug_vis',
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=3.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.05, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.5),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=4.0),
            pc_range=[-30.0, -15.0, -10.0, 30.0, 15.0, 10.0],
        )
    )),
)

# Data loading optimization
data = dict(
    workers_per_gpu=8,  # Speed up data loading
)

# CRITICAL: Disable evaluation to avoid crash
evaluation = dict(interval=999)

# Training settings
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

optimizer = dict(
    type='AdamW',
    lr=1e-3,  #(2e-4) Can try slightly higher than Stage 1 since fewer params training
    weight_decay=0.01,
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# Load from Stage 1 checkpoint
load_from = 'work_dirs/stage3_seg_full/epoch_2.pth'  # ← Your best seg checkpoint
resume_from = None

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')  # View smoothed curves
    ]
)

work_dir = 'work_dirs/stage4_maptr_frozen_seg_mini'