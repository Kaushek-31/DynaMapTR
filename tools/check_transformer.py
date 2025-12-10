#!/usr/bin/env python
"""
Diagnostic script to check available transformers in your codebase.

Run this to determine which config to use.
"""

import sys
import os

def check_transformers():
    """Check which transformers are available in the codebase."""
    
    print("="*60)
    print("TRANSFORMER DIAGNOSTIC TOOL")
    print("="*60)
    print()
    
    # Try to import and check registry
    try:
        # Try multiple import paths for transformer registry
        transformer_registry = None
        registry_source = None
        
        # Try mmdet3d first
        try:
            from mmdet3d.models.builder import MODELS
            transformer_registry = MODELS
            registry_source = "mmdet3d.models.builder.MODELS"
        except:
            pass
        
        # Try mmdet
        if transformer_registry is None:
            try:
                from mmdet.models.builder import MODELS
                transformer_registry = MODELS
                registry_source = "mmdet.models.builder.MODELS"
            except:
                pass
        
        # Try custom plugin
        if transformer_registry is None:
            try:
                # Add project root to path
                sys.path.insert(0, os.getcwd())
                from projects.mmdet3d_plugin.models.builder import TRANSFORMER
                transformer_registry = TRANSFORMER
                registry_source = "projects.mmdet3d_plugin"
            except:
                pass
        
        if transformer_registry is None:
            print("✗ Could not import transformer registry from any source")
            print()
            print("Tried:")
            print("  - mmdet3d.models.builder.MODELS")
            print("  - mmdet.models.builder.MODELS")
            print("  - projects.mmdet3d_plugin")
            print()
            print("This usually means:")
            print("  1. MMDetection3D is not properly installed")
            print("  2. Plugin directory not properly set up")
            print("  3. Wrong Python environment")
            print()
            return False
        
        print(f"✓ Successfully imported registry from {registry_source}")
        print()
        
        # List all registered transformers
        try:
            registered = list(transformer_registry.module_dict.keys())
            
            # Filter for transformers
            transformer_names = [name for name in registered if 'transformer' in name.lower() or 'perception' in name.lower()]
            
            if transformer_names:
                print(f"Found {len(transformer_names)} transformer-related modules:")
                for name in sorted(transformer_names):
                    print(f"  • {name}")
            else:
                print(f"Found {len(registered)} registered modules")
                print("No transformer-specific modules found.")
                print()
                print("Showing all modules (first 20):")
                for name in sorted(registered)[:20]:
                    print(f"  • {name}")
            print()
            
            # Check for specific transformers we need
            target_transformers = [
                'MapTRPerceptionTransformer',
                'PerceptionTransformer',
                'BEVFormerMapTRTransformer',
            ]
            
            print("Checking for required transformers:")
            available = {}
            for trans in target_transformers:
                exists = trans in registered
                status = "✓ FOUND" if exists else "✗ NOT FOUND"
                print(f"  {trans}: {status}")
                available[trans] = exists
            print()
            
            # Provide recommendation
            print("="*60)
            print("RECOMMENDATION:")
            print("="*60)
            
            if available['MapTRPerceptionTransformer']:
                print("✓ Use: bevformer_maptr_seg_mask.py (main config)")
                print("  Transformer type: 'MapTRPerceptionTransformer'")
                print()
            elif available['PerceptionTransformer']:
                print("✓ Use: bevformer_maptr_seg_mask_fallback.py")
                print("  Transformer type: 'PerceptionTransformer'")
                print()
            else:
                print("⚠ WARNING: Required transformers not found!")
                print("  This is normal if transformers aren't registered yet.")
                print("  Recommendation:")
                print("  1. If you have MapTR code: try main config")
                print("  2. If you have BEVFormer only: try fallback config")
                print("  3. The transformer will be registered when you import the modules")
                print()
        except Exception as e:
            print(f"✗ Error accessing registry: {e}")
            print()
            return False
        
    except ImportError as e:
        print(f"✗ Error importing registry: {e}")
        print()
        print("This usually means:")
        print("  1. MMDetection3D is not properly installed")
        print("  2. Or wrong Python environment")
        print()
        return False
    
    # Check for transformer files
    print("="*60)
    print("CHECKING FILESYSTEM:")
    print("="*60)
    print()
    
    plugin_dir = 'projects/mmdet3d_plugin'
    if os.path.exists(plugin_dir):
        print(f"✓ Found plugin directory: {plugin_dir}")
        print()
        
        # Search for transformer files
        transformer_files = []
        for root, dirs, files in os.walk(plugin_dir):
            for file in files:
                if 'transformer' in file.lower() and file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    transformer_files.append(full_path)
        
        if transformer_files:
            print(f"Found {len(transformer_files)} transformer file(s):")
            for f in transformer_files:
                print(f"  • {f}")
            print()
        else:
            print("⚠ No transformer files found")
            print()
        
        # Check for specific directories
        subdirs = ['bevformer', 'maptr', 'bevformer_maptr']
        print("Plugin subdirectories:")
        for subdir in subdirs:
            path = os.path.join(plugin_dir, subdir)
            exists = os.path.exists(path)
            status = "✓" if exists else "✗"
            print(f"  {status} {subdir}/")
        print()
        
    else:
        print(f"✗ Plugin directory not found: {plugin_dir}")
        print("  Make sure you're running from project root")
        print()
    
    # Check imports
    print("="*60)
    print("CHECKING IMPORTS:")
    print("="*60)
    print()
    
    init_file = os.path.join(plugin_dir, '__init__.py')
    if os.path.exists(init_file):
        print(f"✓ Found {init_file}")
        with open(init_file, 'r') as f:
            content = f.read()
            
        # Check for relevant imports
        imports = [
            'bevformer',
            'maptr',
            'transformer',
        ]
        
        print("Checking for imports:")
        for imp in imports:
            if imp in content.lower():
                print(f"  ✓ {imp}")
            else:
                print(f"  ✗ {imp}")
        print()
    else:
        print(f"✗ __init__.py not found in {plugin_dir}")
        print()
    
    return True


def main():
    """Main function."""
    print()
    try:
        success = check_transformers()
        
        print()
        print("="*60)
        print("NEXT STEPS:")
        print("="*60)
        print()
        
        if success:
            print("1. Choose config based on what you have:")
            print("   - bevformer_maptr_seg_mask.py (main)")
            print("   - bevformer_maptr_seg_mask_fallback.py (fallback)")
            print()
            print("2. If issues persist:")
            print("   - Check that plugin imports are set up")
            print("   - Verify transformer files exist")
            print("   - See TROUBLESHOOTING_TRANSFORMER.md")
            print()
            print("3. Test config loading:")
            print("   python -c \"from mmcv import Config; Config.fromfile('configs/bevformer_maptr_seg_mask.py')\"")
            print()
        else:
            print("1. Install/verify MMDetection3D installation")
            print("2. Check Python environment")
            print("3. Verify you're in correct directory")
            print()
        
        return 0 if success else 1
            
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())