import sys
sys.path.insert(0, '.')
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
import sys
import os
import projects.mmdet3d_plugin

def verify_weight_loading(config_path, checkpoint_path):
    print(f"\n1. Reading Config: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Disable pretrained warnings for this test
    cfg.model.pretrained = None 
    
    print("2. Building Model (You will see the 'Initialized...' logs here)...")
    # This triggers the logs you showed me
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    # --- CHECK 1: PICK A SPECIFIC LAYER TO MONITOR ---
    # We choose a layer that we know is "Initialized by user-defined" in your logs
    # e.g., pts_bbox_head.transformer.level_embeds
    target_layer_name = 'pts_bbox_head.transformer.level_embeds'
    
    try:
        # Get the random initialization value
        initial_weight = model.pts_bbox_head.transformer.level_embeds.data.clone()
        print(f"\n[Status] Model built. 'level_embeds' first 5 values (Random Init):\n {initial_weight[0, :5]}")
    except AttributeError:
        print(f"Error: Could not find layer {target_layer_name}. Check model structure.")
        return

    print(f"\n3. Loading Checkpoint manually from: {checkpoint_path}")
    # This simulates what 'load_from' does internally
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Get the weight AFTER loading
    loaded_weight = model.pts_bbox_head.transformer.level_embeds.data.clone()
    print(f"[Status] Checkpoint loaded. 'level_embeds' first 5 values (After Load):\n {loaded_weight[0, :5]}")

    # --- VERIFICATION LOGIC ---
    print("\n" + "="*40)
    print("       VERIFICATION RESULTS       ")
    print("="*40)

    # TEST A: Did the weights change at all?
    if not torch.equal(initial_weight, loaded_weight):
        print("✅ SUCCESS: The weights CHANGED after loading the checkpoint.")
    else:
        print("❌ FAILURE: The weights are IDENTICAL before and after loading.")
        print("   (This means the checkpoint was NOT applied to this layer).")

    # TEST B: Do they match the file exactly?
    # We inspect the raw checkpoint file dictionary
    if 'state_dict' in checkpoint:
        ckpt_dict = checkpoint['state_dict']
    else:
        ckpt_dict = checkpoint # Sometimes it's the raw dict
        
    # Note: State dict keys usually start with 'pts_bbox_head...', but sometimes have 'module.' prefix
    # We search for the key in the file
    key_found = False
    for k in ckpt_dict.keys():
        if target_layer_name in k:
            file_weight = ckpt_dict[k].cpu()
            key_found = True
            
            if torch.allclose(loaded_weight, file_weight):
                print(f"✅ SUCCESS: Model weights MATCH the '{k}' in the checkpoint file exactly.")
            else:
                print(f"❌ FAILURE: Mismatch between model and file for '{k}'.")
            break
            
    if not key_found:
        print(f"⚠️ WARNING: Could not find key '{target_layer_name}' in the checkpoint file.")
        print("   This might mean your checkpoint has a different architecture or prefix (e.g., 'module.').")

if __name__ == '__main__':
    # Usage: python verify_weights.py path/to/config.py path/to/checkpoint.pth
    
    # HARDCODED PATHS FOR YOUR CONVENIENCE (Edit these or pass as args)
    cfg_path = 'projects/configs/custom/bevformer_maptr_frozen_seg.py' # <--- CHANGE THIS
    ckpt_path = 'work_dirs/stage3_seg_full/epoch_2.pth'
    
    if len(sys.argv) > 2:
        cfg_path = sys.argv[1]
        ckpt_path = sys.argv[2]
        
    if not os.path.exists(cfg_path):
        print(f"Please update the 'cfg_path' in the script. Could not find: {cfg_path}")
    else:
        verify_weight_loading(cfg_path, ckpt_path)