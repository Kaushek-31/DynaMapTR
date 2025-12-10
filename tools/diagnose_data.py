# Save as tools/debug_data_flow.py
import sys
sys.path.insert(0, '.')

from mmcv import Config
import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataset
from torch.utils.data import DataLoader

print("\n" + "="*80)
print("DEBUGGING DATA FLOW")
print("="*80)

# Build dataset
cfg = Config.fromfile('projects/configs/bevformer/bevformer_small_seg_maptr.py')
dataset_cfg = cfg.data.val.copy()
dataset_cfg.pop('samples_per_gpu', None)
dataset_cfg.pop('workers_per_gpu', None)
dataset_cfg.test_mode = True

dataset = build_dataset(dataset_cfg)

# Method 1: Direct indexing
print("\n--- METHOD 1: Direct dataset[0] ---")
data = dataset[0]
print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys())}")

if 'img_metas' in data:
    im = data['img_metas']
    print(f"\nimg_metas: {type(im)}")
    print(f"  Length: {len(im)}")
    print(f"  img_metas[0]: {type(im[0])}")
    if hasattr(im[0], 'data'):
        print(f"  img_metas[0].data: {type(im[0].data)}")
        if isinstance(im[0].data, dict):
            print(f"  Keys: {list(im[0].data.keys())[:10]}")

# Method 2: Using DataLoader with default collate
print("\n--- METHOD 2: DataLoader with default collate_fn ---")
from mmcv.parallel import DataContainer
from torch.utils.data.dataloader import default_collate

# Try the actual collate_fn from mmdet
try:
    from mmdet3d.datasets import build_dataloader
    
    # Build with mmdet's dataloader
    print("\nTrying mmdet3d's build_dataloader...")
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    
    # Get first batch
    for i, data in enumerate(data_loader):
        print(f"\nBatch from mmdet3d dataloader:")
        print(f"  Type: {type(data)}")
        print(f"  Keys: {list(data.keys())}")
        
        if 'img_metas' in data:
            im = data['img_metas']
            print(f"\n  img_metas: {type(im)}")
            if isinstance(im, list):
                print(f"    Length: {len(im)}")
                if len(im) > 0:
                    print(f"    img_metas[0]: {type(im[0])}")
                    if isinstance(im[0], list):
                        print(f"      Length: {len(im[0])}")
                        if len(im[0]) > 0:
                            print(f"      img_metas[0][0]: {type(im[0][0])}")
                            if isinstance(im[0][0], dict):
                                print(f"      Keys: {list(im[0][0].keys())[:10]}")
        break
        
except Exception as e:
    print(f"Error with mmdet3d dataloader: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)