# Save as: tools/test_first_batch_timeout.py
"""Test first batch with timeout to see where it hangs."""
import sys
sys.path.insert(0, '.')

import torch
import signal
from mmcv import Config
from mmdet3d.datasets import build_dataset
import projects.mmdet3d_plugin

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out!")

def main():
    print("Loading config...")
    cfg = Config.fromfile('projects/configs/bevformer/bevformer_small_seg_maptr.py')
    
    cfg.data.workers_per_gpu = 0
    cfg.data.samples_per_gpu = 1
    
    print("Building dataset...")
    dataset = build_dataset(cfg.data.train)
    print(f"✓ Dataset: {len(dataset)} samples")
    
    print("\nTesting direct dataset access (with timeout)...")
    
    # Set 30 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)
    
    try:
        print("Attempting to load sample 0...")
        data = dataset[0]
        signal.alarm(0)  # Cancel timeout
        print(f"✓ Got sample 0!")
        print(f"  Keys: {data.keys()}")
    except TimeoutException:
        print("✗ TIMEOUT after 30 seconds!")
        print("The dataset.__getitem__ is hanging.")
        return
    except Exception as e:
        signal.alarm(0)
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("Now testing with DataLoader...")
    print("="*80)
    
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    print("✓ DataLoader built")
    
    print("\nFetching first batch...")
    signal.alarm(30)
    
    try:
        for i, data_batch in enumerate(data_loader):
            signal.alarm(0)
            print(f"✓ Got batch {i}")
            break
    except TimeoutException:
        print("✗ TIMEOUT in dataloader after 30 seconds!")
        return
    
    print("\n✓ SUCCESS!")

if __name__ == '__main__':
    main()