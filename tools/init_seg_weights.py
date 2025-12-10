#!/usr/bin/env python
"""
Initialize segmentation weights and combine with existing BEVFormer checkpoint.

This script:
1. Loads existing BEVFormer weights (encoder, backbone, etc.)
2. Initializes segmentation head with proper weights
3. Combines them into a single checkpoint ready for training

Usage:
    python tools/init_seg_weights.py \
        --base-checkpoint ckpts/bevformer_small_epoch_24.pth \
        --output ckpts/bevformer_maptr_init.pth \
        --seg-classes 4 \
        --bev-h 200 \
        --bev-w 100 \
        --seg-size 200 400 \
        --verify
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import os
import sys


def initialize_segencode_weights(seg_classes=4, in_channels=256, bev_h=200, bev_w=100, 
                                  seg_h=200, seg_w=400, verbose=True):
    """
    Initialize SegEncode module weights matching the architecture.
    
    SegEncode architecture:
    - conv1: Conv2d(256, 64, kernel_size=7, stride=2, padding=3)
    - bn1: BatchNorm2d(64)
    - layer1: ResNet BasicBlock (64 channels)
    - layer2: ResNet BasicBlock (128 channels)
    - layer3: ResNet BasicBlock (256 channels)
    - up1: Up(64+256, 256, scale_factor=4)
    - up2: Upsampling + Conv(256, 128) + Conv(128, seg_classes)
    """
    from torchvision.models.resnet import resnet18, BasicBlock
    
    if verbose:
        print(f"\nInitializing SegEncode with:")
        print(f"  Input channels: {in_channels}")
        print(f"  Output classes: {seg_classes}")
        print(f"  BEV size: {bev_h}x{bev_w}")
        print(f"  Seg GT size: {seg_h}x{seg_w}")
    
    state_dict = OrderedDict()
    
    # Get ResNet18 pretrained weights for encoder
    resnet = resnet18(pretrained=True)
    
    # conv1 (7x7 conv, in_channels -> 64)
    # We need to adapt from 256 channels to 64
    conv1_weight = torch.randn(64, in_channels, 7, 7) * 0.01
    nn.init.kaiming_normal_(conv1_weight, mode='fan_out', nonlinearity='relu')
    state_dict['pts_bbox_head.seg_head.conv1.weight'] = conv1_weight
    
    # bn1
    state_dict['pts_bbox_head.seg_head.bn1.weight'] = resnet.bn1.weight.clone()
    state_dict['pts_bbox_head.seg_head.bn1.bias'] = resnet.bn1.bias.clone()
    state_dict['pts_bbox_head.seg_head.bn1.running_mean'] = resnet.bn1.running_mean.clone()
    state_dict['pts_bbox_head.seg_head.bn1.running_var'] = resnet.bn1.running_var.clone()
    state_dict['pts_bbox_head.seg_head.bn1.num_batches_tracked'] = resnet.bn1.num_batches_tracked.clone()
    
    # layer1 (ResNet blocks)
    for name, param in resnet.layer1.named_parameters():
        state_dict[f'pts_bbox_head.seg_head.layer1.{name}'] = param.clone()
    for name, buffer in resnet.layer1.named_buffers():
        state_dict[f'pts_bbox_head.seg_head.layer1.{name}'] = buffer.clone()
    
    # layer2 (ResNet blocks)
    for name, param in resnet.layer2.named_parameters():
        state_dict[f'pts_bbox_head.seg_head.layer2.{name}'] = param.clone()
    for name, buffer in resnet.layer2.named_buffers():
        state_dict[f'pts_bbox_head.seg_head.layer2.{name}'] = buffer.clone()
    
    # layer3 (ResNet blocks)
    for name, param in resnet.layer3.named_parameters():
        state_dict[f'pts_bbox_head.seg_head.layer3.{name}'] = param.clone()
    for name, buffer in resnet.layer3.named_buffers():
        state_dict[f'pts_bbox_head.seg_head.layer3.{name}'] = buffer.clone()
    
    # up1 - Up module (64+256 -> 256)
    # up1.up is nn.Upsample (no parameters)
    # up1.conv is Sequential with 2 convs
    
    # up1.conv.0: Conv2d(320, 256, kernel_size=3, padding=1)
    conv_weight = torch.randn(256, 320, 3, 3) * 0.01
    nn.init.kaiming_normal_(conv_weight, mode='fan_out', nonlinearity='relu')
    state_dict['pts_bbox_head.seg_head.up1.conv.0.weight'] = conv_weight
    state_dict['pts_bbox_head.seg_head.up1.conv.1.weight'] = torch.ones(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.1.bias'] = torch.zeros(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.1.running_mean'] = torch.zeros(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.1.running_var'] = torch.ones(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.1.num_batches_tracked'] = torch.tensor(0)
    
    # up1.conv.3: Conv2d(256, 256, kernel_size=3, padding=1)
    conv_weight = torch.randn(256, 256, 3, 3) * 0.01
    nn.init.kaiming_normal_(conv_weight, mode='fan_out', nonlinearity='relu')
    state_dict['pts_bbox_head.seg_head.up1.conv.3.weight'] = conv_weight
    state_dict['pts_bbox_head.seg_head.up1.conv.4.weight'] = torch.ones(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.4.bias'] = torch.zeros(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.4.running_mean'] = torch.zeros(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.4.running_var'] = torch.ones(256)
    state_dict['pts_bbox_head.seg_head.up1.conv.4.num_batches_tracked'] = torch.tensor(0)
    
    # up2 - Final upsampling and classification
    # up2.0 is nn.Upsample (no parameters)
    # up2.1: Conv2d(256, 128, kernel_size=3, padding=1)
    conv_weight = torch.randn(128, 256, 3, 3) * 0.01
    nn.init.kaiming_normal_(conv_weight, mode='fan_out', nonlinearity='relu')
    state_dict['pts_bbox_head.seg_head.up2.1.weight'] = conv_weight
    state_dict['pts_bbox_head.seg_head.up2.2.weight'] = torch.ones(128)
    state_dict['pts_bbox_head.seg_head.up2.2.bias'] = torch.zeros(128)
    state_dict['pts_bbox_head.seg_head.up2.2.running_mean'] = torch.zeros(128)
    state_dict['pts_bbox_head.seg_head.up2.2.running_var'] = torch.ones(128)
    state_dict['pts_bbox_head.seg_head.up2.2.num_batches_tracked'] = torch.tensor(0)
    
    # up2.4: Conv2d(128, seg_classes, kernel_size=1, padding=0)
    conv_weight = torch.randn(seg_classes, 128, 1, 1) * 0.01
    nn.init.kaiming_normal_(conv_weight, mode='fan_out', nonlinearity='relu')
    state_dict['pts_bbox_head.seg_head.up2.4.weight'] = conv_weight
    state_dict['pts_bbox_head.seg_head.up2.4.bias'] = torch.zeros(seg_classes)
    
    # up_sampler - Upsample (no parameters)
    
    if verbose:
        print(f"  ✓ Initialized {len(state_dict)} segmentation parameters")
    
    return state_dict


def initialize_feat_cropper_weights(verbose=True):
    """
    Initialize BevFeatureSlicer (feat_cropper) weights.
    Note: This module typically has no learnable parameters, only buffers.
    """
    state_dict = OrderedDict()
    
    # BevFeatureSlicer only has grid buffers, which are computed at runtime
    # No parameters to initialize
    
    if verbose:
        print(f"  ✓ feat_cropper has no learnable parameters (grid computed at runtime)")
    
    return state_dict


def combine_checkpoints(base_checkpoint_path, output_path, 
                       seg_classes=4, in_channels=256,
                       bev_h=200, bev_w=100, 
                       seg_h=200, seg_w=400,
                       exclude_keys=None,
                       verbose=True):
    """
    Combine base BEVFormer checkpoint with initialized segmentation weights.
    """
    if exclude_keys is None:
        # Default keys to exclude (shape-sensitive parameters)
        exclude_keys = [
            'code_weights',
            'pts_bbox_head.code_weights',
        ]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"COMBINING BEVFORMER + SEGMENTATION WEIGHTS")
        print(f"{'='*70}")
        print(f"\nLoading base checkpoint: {base_checkpoint_path}")
    
    # Load base checkpoint
    base_checkpoint = torch.load(base_checkpoint_path, map_location='cpu')
    
    if 'state_dict' in base_checkpoint:
        base_state_dict = base_checkpoint['state_dict']
    else:
        base_state_dict = base_checkpoint
    
    if verbose:
        print(f"  ✓ Loaded {len(base_state_dict)} parameters from base checkpoint")
    
    # Filter out excluded keys
    excluded = []
    filtered_base_state_dict = OrderedDict()
    for key, value in base_state_dict.items():
        if any(exclude_key in key for exclude_key in exclude_keys):
            excluded.append(f"{key} (shape: {list(value.shape)})")
        else:
            filtered_base_state_dict[key] = value
    
    if excluded and verbose:
        print(f"\n  Excluded {len(excluded)} shape-sensitive parameters:")
        for item in excluded:
            print(f"    ✗ {item}")
    
    # Initialize segmentation weights
    if verbose:
        print(f"\nInitializing segmentation weights...")
    
    seg_state_dict = initialize_segencode_weights(
        seg_classes=seg_classes,
        in_channels=in_channels,
        bev_h=bev_h,
        bev_w=bev_w,
        seg_h=seg_h,
        seg_w=seg_w,
        verbose=verbose
    )
    
    feat_cropper_dict = initialize_feat_cropper_weights(verbose=verbose)
    
    # Combine state dicts
    combined_state_dict = OrderedDict()
    
    # Add filtered base weights
    for key, value in filtered_base_state_dict.items():
        combined_state_dict[key] = value
    
    # Add segmentation weights (will overwrite if any conflicts)
    for key, value in seg_state_dict.items():
        combined_state_dict[key] = value
    
    for key, value in feat_cropper_dict.items():
        combined_state_dict[key] = value
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Combined checkpoint statistics:")
        print(f"  Base parameters (after filtering): {len(filtered_base_state_dict)}")
        print(f"  Excluded parameters: {len(excluded)}")
        print(f"  Segmentation parameters: {len(seg_state_dict)}")
        print(f"  Total parameters: {len(combined_state_dict)}")
        
        # Count parameters by module
        module_counts = {}
        for key in combined_state_dict.keys():
            if '.' in key:
                module = '.'.join(key.split('.')[:2])
            else:
                module = 'other'
            module_counts[module] = module_counts.get(module, 0) + 1
        
        print(f"\n  Parameters by module:")
        for module, count in sorted(module_counts.items()):
            marker = "← NEW" if 'seg' in module else ""
            print(f"    {module:40s}: {count:4d} params {marker}")
    
    # Create output checkpoint
    output_checkpoint = {
        'state_dict': combined_state_dict,
        'meta': {
            'description': 'BEVFormer + initialized segmentation weights',
            'base_checkpoint': os.path.basename(base_checkpoint_path),
            'seg_classes': seg_classes,
            'bev_size': (bev_h, bev_w),
            'seg_gt_size': (seg_h, seg_w),
            'total_params': len(combined_state_dict),
            'excluded_keys': excluded,
        }
    }
    
    # Copy original meta if available
    if 'meta' in base_checkpoint:
        output_checkpoint['base_meta'] = base_checkpoint['meta']
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(output_checkpoint, output_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ CHECKPOINT SAVED")
        print(f"{'='*70}")
        print(f"  Output: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"{'='*70}")
    
    return True


def verify_combined_checkpoint(checkpoint_path, seg_classes=4):
    """Verify the combined checkpoint."""
    print(f"\n{'='*70}")
    print(f"VERIFYING COMBINED CHECKPOINT")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}\n")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"  ✓ Loaded {len(state_dict)} parameters")
        
        # Check for base components
        base_components = ['img_backbone', 'img_neck', 'pts_bbox_head.transformer']
        found_base = []
        for component in base_components:
            if any(component in key for key in state_dict.keys()):
                found_base.append(component)
        
        if found_base:
            print(f"  ✓ Base components found: {found_base}")
        else:
            print(f"  ✗ WARNING: No base components found!")
            return False
        
        # Check for segmentation components
        seg_components = ['seg_head.conv1', 'seg_head.layer1', 'seg_head.up1', 'seg_head.up2']
        found_seg = []
        for component in seg_components:
            if any(component in key for key in state_dict.keys()):
                found_seg.append(component)
        
        if found_seg:
            print(f"  ✓ Segmentation components found: {found_seg}")
        else:
            print(f"  ✗ WARNING: Segmentation components not found!")
            return False
        
        # Check BEV embeddings
        bev_keys = [k for k in state_dict.keys() if 'bev_embedding' in k]
        if bev_keys:
            print(f"  ✓ BEV embeddings found: {len(bev_keys)} parameters")
        
        # Check metadata
        if 'meta' in checkpoint:
            print(f"\n  Metadata:")
            for key, value in checkpoint['meta'].items():
                print(f"    {key}: {value}")
        
        # Verify seg_head output layer has correct number of classes
        output_layer_key = 'pts_bbox_head.seg_head.up2.4.weight'
        if output_layer_key in state_dict:
            output_shape = state_dict[output_layer_key].shape
            actual_classes = output_shape[0]
            if actual_classes == seg_classes:
                print(f"\n  ✓ Output layer matches seg_classes: {actual_classes} classes")
            else:
                print(f"\n  ✗ WARNING: Output layer mismatch!")
                print(f"      Expected: {seg_classes} classes")
                print(f"      Got: {actual_classes} classes")
                return False
        
        print(f"\n{'='*70}")
        print(f"✅ CHECKPOINT IS VALID")
        print(f"{'='*70}")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_checkpoint_keys(checkpoint_path, pattern=None):
    """List keys in checkpoint."""
    print(f"\nListing keys in: {checkpoint_path}")
    print(f"{'-'*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    keys = list(state_dict.keys())
    
    if pattern:
        keys = [k for k in keys if pattern.lower() in k.lower()]
        print(f"Keys matching '{pattern}' ({len(keys)} found):")
    else:
        print(f"All keys ({len(keys)} total):")
    
    for key in keys:
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
        print(f"  {key:60s} {str(list(shape) if hasattr(shape, '__iter__') else shape):20s}")
    
    if len(keys) == 0:
        print("  (no matching keys found)")
    
    print(f"{'-'*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Initialize segmentation weights and combine with BEVFormer checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create combined checkpoint with default settings
  python tools/init_seg_weights.py \\
      --base-checkpoint ckpts/bevformer_small_epoch_24.pth \\
      --output ckpts/bevformer_maptr_init.pth \\
      --verify

  # Customize segmentation settings
  python tools/init_seg_weights.py \\
      --base-checkpoint ckpts/bevformer_small_epoch_24.pth \\
      --output ckpts/bevformer_maptr_init.pth \\
      --seg-classes 4 \\
      --bev-h 200 --bev-w 100 \\
      --seg-size 200 400 \\
      --verify

  # List keys in existing checkpoint
  python tools/init_seg_weights.py \\
      --base-checkpoint ckpts/bevformer_small_epoch_24.pth \\
      --list-keys \\
      --pattern seg
        """)
    
    parser.add_argument(
        '--base-checkpoint',
        type=str,
        required=True,
        help='Path to base BEVFormer checkpoint')
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save combined checkpoint')
    
    parser.add_argument(
        '--seg-classes',
        type=int,
        default=4,
        help='Number of segmentation classes (default: 4)')
    
    parser.add_argument(
        '--in-channels',
        type=int,
        default=256,
        help='BEV feature channels (default: 256)')
    
    parser.add_argument(
        '--bev-h',
        type=int,
        default=200,
        help='BEV height (default: 200)')
    
    parser.add_argument(
        '--bev-w',
        type=int,
        default=100,
        help='BEV width (default: 100)')
    
    parser.add_argument(
        '--seg-size',
        type=int,
        nargs=2,
        default=[200, 400],
        metavar=('H', 'W'),
        help='Segmentation GT size (H W) (default: 200 400)')
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the combined checkpoint after creation')
    
    parser.add_argument(
        '--list-keys',
        action='store_true',
        help='List keys in base checkpoint (no combination)')
    
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='Filter keys by pattern (for --list-keys)')
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity')
    
    parser.add_argument(
        '--exclude-keys',
        type=str,
        nargs='+',
        default=['code_weights'],
        help='Keys to exclude from base checkpoint (default: code_weights)')
        
    args = parser.parse_args()
    
    # List keys mode
    if args.list_keys:
        list_checkpoint_keys(args.base_checkpoint, args.pattern)
        return 0
    
    # Combine mode - require output path
    if not args.output:
        parser.error("--output is required when not using --list-keys")
    
    # Combine checkpoints
    success = combine_checkpoints(
        base_checkpoint_path=args.base_checkpoint,
        output_path=args.output,
        seg_classes=args.seg_classes,
        in_channels=args.in_channels,
        bev_h=args.bev_h,
        bev_w=args.bev_w,
        seg_h=args.seg_size[0],
        seg_w=args.seg_size[1],
        exclude_keys=args.exclude_keys,
        verbose=not args.quiet,
    )
    
    if not success:
        print("\n❌ Combination failed!")
        return 1
    
    # Verify if requested
    if args.verify:
        if not verify_combined_checkpoint(args.output, args.seg_classes):
            print("\n⚠️  Verification failed!")
            return 1
    
    print("\n✅ All done! Your checkpoint is ready for training.")
    print(f"\n   Update your config: load_from = '{args.output}'")
    return 0


if __name__ == '__main__':
    sys.exit(main())