#!/usr/bin/env python
"""
Extract segmentation weights from a full BEVFormer checkpoint.

Usage:
    python tools/extract_seg_weights.py \
        --checkpoint work_dirs/bevformer_seg/epoch_24.pth \
        --output ckpts/bevformer_seg_pretrained.pth \
        --verify
"""

import argparse
import torch
from collections import OrderedDict
import os

# Add to seg_patterns exclusion list
EXCLUDE_KEYS = [
    'code_weights',  # Shape depends on config
]

def extract_segmentation_weights(checkpoint_path, output_path, verbose=True):
    """Extract segmentation-related weights from a checkpoint."""
    if verbose:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if verbose:
        print(f"Total parameters in checkpoint: {len(state_dict)}")
        print(f"\nSearching for segmentation parameters...")
    
    # Define segmentation-related key patterns
    seg_patterns = [
        'pts_bbox_head.seg_head',
        'pts_bbox_head.feat_cropper',
        'pts_bbox_head.seg_decoder',
        'pts_bbox_head.seg_encoder',
        # Also include shared components needed for segmentation
        'pts_bbox_head.bev_embedding',
        'pts_bbox_head.positional_encoding',
    ]
    
    # Extract segmentation weights
    seg_state_dict = OrderedDict()
    skipped_keys = []
    
    for key, value in state_dict.items():
        # Check if key matches any segmentation pattern
        if any(pattern in key for pattern in seg_patterns):
            seg_state_dict[key] = value
            if verbose:
                print(f"  ✓ {key:60s} {list(value.shape)}")
        elif 'seg' in key.lower():
            # Catch any other seg-related keys we might have missed
            skipped_keys.append(key)
    for key, value in state_dict.items():
        # Skip excluded keys
        if any(exclude in key for exclude in EXCLUDE_KEYS):
            if verbose:
                print(f"  ✗ EXCLUDED: {key} (shape-sensitive)")
            continue
    # Warn about skipped seg-related keys
    if skipped_keys and verbose:
        print(f"\nWARNING: Found {len(skipped_keys)} seg-related keys that were skipped:")
        for key in skipped_keys[:5]:
            print(f"  ? {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    
    if len(seg_state_dict) == 0:
        print("\n❌ ERROR: No segmentation parameters found!")
        print("\nAvailable keys in checkpoint:")
        for key in sorted(state_dict.keys())[:20]:
            print(f"  - {key}")
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more")
        return False
    
    # Create new checkpoint with segmentation weights
    seg_checkpoint = {
        'state_dict': seg_state_dict,
        'meta': {
            'description': 'Segmentation-only weights for BEVFormer',
            'source_checkpoint': os.path.basename(checkpoint_path),
            'num_params': len(seg_state_dict),
            'seg_classes': 4,  # Update if different
            'extraction_patterns': seg_patterns,
        }
    }
    
    # Copy original meta if available
    if 'meta' in checkpoint:
        seg_checkpoint['original_meta'] = checkpoint['meta']
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(seg_checkpoint, output_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ EXTRACTION SUCCESSFUL")
        print(f"{'='*70}")
        print(f"  Parameters extracted: {len(seg_state_dict):,}")
        print(f"  Output file: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Count parameters by module
        module_counts = {}
        for key in seg_state_dict.keys():
            module = key.split('.')[1] if '.' in key else 'other'
            module_counts[module] = module_counts.get(module, 0) + 1
        
        print(f"\n  Parameters by module:")
        for module, count in sorted(module_counts.items()):
            print(f"    {module:30s}: {count:3d} parameters")
        
        print(f"{'='*70}\n")
    
    return True


def verify_checkpoint(checkpoint_path):
    """Verify the extracted checkpoint can be loaded."""
    print(f"\nVerifying checkpoint: {checkpoint_path}")
    print(f"{'-'*70}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  ✓ state_dict found with {len(state_dict)} parameters")
        else:
            print(f"  ✓ Direct state dict with {len(checkpoint)} parameters")
            state_dict = checkpoint
        
        if 'meta' in checkpoint:
            print(f"  ✓ Metadata:")
            for key, value in checkpoint['meta'].items():
                print(f"      {key}: {value}")
        
        # Check for required keys
        required_patterns = ['seg_head']
        found_modules = set()
        for key in state_dict.keys():
            for pattern in required_patterns:
                if pattern in key:
                    found_modules.add(pattern)
        
        if found_modules:
            print(f"  ✓ Found required modules: {found_modules}")
        else:
            print(f"  ✗ WARNING: Required segmentation modules not found!")
            print(f"      Available keys: {list(state_dict.keys())[:5]}")
            return False
        
        # Try to count total parameters
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  ✓ Total parameters: {total_params:,}")
        
        print(f"\n  ✅ Checkpoint is valid and ready to use!")
        print(f"{'-'*70}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_checkpoint_keys(checkpoint_path, pattern=None):
    """List all keys in a checkpoint, optionally filtered by pattern."""
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
        print(f"Keys matching pattern '{pattern}' ({len(keys)} found):")
    else:
        print(f"All keys ({len(keys)} total):")
    
    for key in keys:
        shape = state_dict[key].shape
        dtype = state_dict[key].dtype
        print(f"  {key:60s} {list(shape):20s} {dtype}")
    
    if len(keys) == 0:
        print("  (no matching keys found)")
    
    print(f"{'-'*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract segmentation weights from BEVFormer checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and verify
  python tools/extract_seg_weights.py \\
      --checkpoint work_dirs/bevformer_seg/epoch_24.pth \\
      --output ckpts/bevformer_seg_pretrained.pth \\
      --verify

  # List all segmentation-related keys
  python tools/extract_seg_weights.py \\
      --checkpoint work_dirs/bevformer_seg/epoch_24.pth \\
      --list-keys \\
      --pattern seg
        """)
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to full BEVFormer checkpoint')
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save segmentation-only checkpoint')
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the extracted checkpoint after creation')
    
    parser.add_argument(
        '--list-keys',
        action='store_true',
        help='List all keys in the checkpoint (no extraction)')
    
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='Filter keys by pattern (for --list-keys)')
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # List keys mode
    if args.list_keys:
        list_checkpoint_keys(args.checkpoint, args.pattern)
        return 0
    
    # Extract mode - require output path
    if not args.output:
        parser.error("--output is required when not using --list-keys")
    
    # Extract weights
    print(f"\n{'='*70}")
    print(f"BEVFORMER SEGMENTATION WEIGHT EXTRACTION")
    print(f"{'='*70}\n")
    
    success = extract_segmentation_weights(
        args.checkpoint,
        args.output,
        verbose=not args.quiet
    )
    
    if not success:
        print("\n❌ Extraction failed!")
        return 1
    
    # Verify if requested
    if args.verify:
        if not verify_checkpoint(args.output):
            print("\n⚠️  Verification failed!")
            return 1
    
    print("\n✅ All done! You can now use the extracted checkpoint for training.")
    print(f"\n   Add to your config: load_from = '{args.output}'")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())