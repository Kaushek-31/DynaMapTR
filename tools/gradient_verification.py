#!/usr/bin/env python
import torch
import argparse
import sys
import os

def verify_gradients(config_file):
    """Verify which parameters are frozen vs trainable"""
    
    # Set up paths
    sys.path.insert(0, os.getcwd())
    
    # Force plugin import BEFORE anything else
    import projects.mmdet3d_plugin.bevformer
    
    # Now load config
    from mmcv import Config
    cfg = Config.fromfile(config_file)
    
    # Import build_model
    from mmdet3d.models import build_model
    
    # Verify registration
    from mmdet.models import DETECTORS, HEADS
    print(f"\n{'='*80}")
    print(f"REGISTRATION CHECK")
    print(f"{'='*80}")
    print(f"Model type in config: {cfg.model.type}")
    print(f"BEVFormerMapTR registered: {'BEVFormerMapTR' in DETECTORS.module_dict}")
    print(f"BEVFormerMapTRHead registered: {'BEVFormerMapTRHead' in HEADS.module_dict}")
    
    if cfg.model.type not in DETECTORS.module_dict:
        print(f"\nERROR: {cfg.model.type} not found in registry!")
        print(f"Available BEVFormer detectors: {[k for k in DETECTORS.module_dict.keys() if 'BEVFormer' in k]}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"BUILDING MODEL")
    print(f"{'='*80}")
    
    # Build model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.train()
    
    print(f"✓ Model built successfully: {type(model).__name__}")
    
    # Collect parameters
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))
    
    # Print trainable parameters
    print(f"\n{'='*80}")
    print(f"TRAINABLE PARAMETERS ({len(trainable_params)} layers, "
          f"{sum(p for _, p in trainable_params):,} params)")
    print(f"{'='*80}")
    for name, numel in trainable_params[:30]:  # Show first 30
        print(f"  ✓ {name:70s} {numel:>12,}")
    if len(trainable_params) > 30:
        print(f"  ... and {len(trainable_params)-30} more layers")
    
    # Print frozen parameters
    print(f"\n{'='*80}")
    print(f"FROZEN PARAMETERS ({len(frozen_params)} layers, "
          f"{sum(p for _, p in frozen_params):,} params)")
    print(f"{'='*80}")
    for name, numel in frozen_params[:30]:  # Show first 30
        print(f"  ✗ {name:70s} {numel:>12,}")
    if len(frozen_params) > 30:
        print(f"  ... and {len(frozen_params)-30} more layers")
    
    # Verify expected components
    expected_trainable = [
        ('transformer.decoder', 'MapTR decoder layers'),
        ('cls_branches', 'Classification heads'),
        ('reg_branches', 'Regression heads'),
        ('instance_embedding', 'Instance query embeddings'),
        ('pts_embedding', 'Point query embeddings')
    ]
    
    expected_frozen = [
        ('transformer.encoder', 'BEVFormer encoder'),
        ('img_backbone', 'Image backbone'),
        ('img_neck', 'Image neck'),
        ('seg_decoder', 'Segmentation decoder')
    ]
    
    print(f"\n{'='*80}")
    print("VERIFICATION CHECKS:")
    print(f"{'='*80}")
    
    print("\nExpected TRAINABLE components:")
    for component, desc in expected_trainable:
        trainable_count = sum(1 for name, _ in trainable_params if component in name)
        frozen_count = sum(1 for name, _ in frozen_params if component in name)
        
        if trainable_count > 0:
            print(f"  ✓ {desc:40s} - {trainable_count} trainable layers")
        elif frozen_count > 0:
            print(f"  ✗ {desc:40s} - FROZEN (should be trainable!)")
        else:
            print(f"  ? {desc:40s} - NOT FOUND")
    
    print("\nExpected FROZEN components:")
    for component, desc in expected_frozen:
        trainable_count = sum(1 for name, _ in trainable_params if component in name)
        frozen_count = sum(1 for name, _ in frozen_params if component in name)
        
        if frozen_count > 0:
            print(f"  ✓ {desc:40s} - {frozen_count} frozen layers")
        elif trainable_count > 0:
            print(f"  ✗ {desc:40s} - TRAINABLE (should be frozen!)")
        else:
            print(f"  ? {desc:40s} - NOT FOUND")
    
    # Summary
    total_params = sum(p for _, p in trainable_params) + sum(p for _, p in frozen_params)
    trainable_pct = 100 * sum(p for _, p in trainable_params) / total_params if total_params > 0 else 0
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"Total parameters:     {total_params:>15,}")
    print(f"Trainable parameters: {sum(p for _, p in trainable_params):>15,} ({trainable_pct:.2f}%)")
    print(f"Frozen parameters:    {sum(p for _, p in frozen_params):>15,} ({100-trainable_pct:.2f}%)")
    
    return trainable_params, frozen_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify gradient flow in BEVFormer+MapTR model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    verify_gradients(args.config)
