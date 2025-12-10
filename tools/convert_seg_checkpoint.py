#!/usr/bin/env python
"""Convert BEVFormerHead_seg checkpoint to BEVFormerMapTRHead format."""
import torch
from collections import OrderedDict
import argparse

def convert_seg_checkpoint(input_path, output_path, exclude_keys=None):
    """
    Convert checkpoint from BEVFormerHead_seg to BEVFormerMapTRHead.
    
    Key remapping:
    - pts_bbox_head.seg_decoder.* → pts_bbox_head.seg_head.*
    """
    if exclude_keys is None:
        exclude_keys = ['code_weights']
    
    print(f"Loading: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')
    
    state_dict = ckpt.get('state_dict', ckpt)
    
    new_state_dict = OrderedDict()
    remapped = []
    excluded = []
    kept = []
    
    for key, value in state_dict.items():
        # Check if should be excluded
        if any(exc in key for exc in exclude_keys):
            excluded.append(f"{key} (shape: {list(value.shape)})")
            continue
        
        new_key = key
        
        # Remap seg_decoder → seg_head
        if 'pts_bbox_head.seg_decoder.' in key:
            new_key = key.replace('pts_bbox_head.seg_decoder.', 'pts_bbox_head.seg_head.')
            remapped.append(f"{key} → {new_key}")
        
        new_state_dict[new_key] = value
        kept.append(new_key)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CHECKPOINT CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total parameters in source: {len(state_dict)}")
    print(f"Parameters kept: {len(kept)}")
    print(f"Parameters remapped: {len(remapped)}")
    print(f"Parameters excluded: {len(excluded)}")
    
    if remapped:
        print(f"\nRemapped keys:")
        for r in remapped[:10]:
            print(f"  {r}")
        if len(remapped) > 10:
            print(f"  ... and {len(remapped) - 10} more")
    
    if excluded:
        print(f"\nExcluded keys:")
        for e in excluded:
            print(f"  {e}")
    
    # Check for segmentation head keys
    seg_keys = [k for k in new_state_dict.keys() if 'seg_head' in k]
    print(f"\nSegmentation head keys in output: {len(seg_keys)}")
    
    # Save
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = new_state_dict
    else:
        ckpt = {'state_dict': new_state_dict}
    
    # Add conversion metadata
    if 'meta' not in ckpt:
        ckpt['meta'] = {}
    ckpt['meta']['converted_from'] = input_path
    ckpt['meta']['key_remapping'] = 'seg_decoder → seg_head'
    
    torch.save(ckpt, output_path)
    print(f"\n✓ Saved converted checkpoint to: {output_path}")
    print(f"{'='*60}")

def verify_checkpoint(checkpoint_path):
    """Verify the converted checkpoint has expected keys."""
    print(f"\nVerifying: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    
    # Check for key components
    components = {
        'img_backbone': False,
        'img_neck': False,
        'transformer': False,
        'seg_head': False,
        'feat_cropper': False,
        'bev_embedding': False,
    }
    
    for key in state_dict.keys():
        for comp in components:
            if comp in key:
                components[comp] = True
    
    print("\nComponent check:")
    for comp, found in components.items():
        status = "✓" if found else "✗"
        print(f"  {status} {comp}")
    
    # List seg_head keys
    seg_keys = [k for k in state_dict.keys() if 'seg_head' in k]
    if seg_keys:
        print(f"\nSegmentation head parameters ({len(seg_keys)}):")
        for k in seg_keys[:5]:
            print(f"  {k}: {list(state_dict[k].shape)}")
        if len(seg_keys) > 5:
            print(f"  ... and {len(seg_keys) - 5} more")
    
    return all(components.values())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Source checkpoint path')
    parser.add_argument('--output', required=True, help='Output checkpoint path')
    parser.add_argument('--verify', action='store_true', help='Verify after conversion')
    parser.add_argument('--exclude-keys', nargs='+', default=['code_weights'],
                        help='Keys to exclude from conversion')
    args = parser.parse_args()
    
    convert_seg_checkpoint(args.input, args.output, args.exclude_keys)
    
    if args.verify:
        verify_checkpoint(args.output)