_base_ = ['./bevformer_small_seg_maptr.py']

# Stage 1: Train segmentation head only

model = dict(
    pts_bbox_head=dict(
        aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=False,
            seg_classes=3, #11, shivam changes
            feat_down_sample=8, #32, shivam changes
        ),
        seg_mask_classes=[],  # No masking during seg-only training
        freeze_seg=False,
        
        # DISABLE MapTR losses
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.0001),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', loss_weight=0.0001),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0),
        
        # Segmentation loss - use CrossEntropyLoss with strong class weights
        loss_seg=dict(
            type='FocalDiceLossV2',
            class_weights=[0.05, 3.0, 10.0],  # bg gets 0.1, vehicle 2.0, ped 4.0
            gamma=2.0,
            alpha=0.25,
            dice_weight=1.0,
            focal_weight=1.0,
        ),
    ),
    train_cfg=dict(pts=dict(
        debug_vis=True,
        debug_iter=50,
        save_dir = 'work_dirs/stage3_seg_full/debug_vis',
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=0.0001),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=0.0001),
            pc_range=[-30.0, -15.0, -10.0, 30.0, 15.0, 10.0],
        ),
    ))
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
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

load_from = 'work_dirs/stage1_seg_only/epoch_24.pth'
resume_from = None

checkpoint_config = dict(max_keep_ckpts=5, interval=2)
work_dir = 'work_dirs/stage3_seg_full'