"""
RasterizeMapVectors pipeline transform for converting vectorized maps to raster format.
This generates segmentation ground truth from vectorized map annotations.
"""

import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString
from shapely import affinity


@PIPELINES.register_module()
class RasterizeMapVectors:
    """
    Rasterize vectorized map annotations to create segmentation ground truth.
    
    This transform converts vector map elements (lines) into rasterized BEV segmentation masks.
    Used for auxiliary segmentation supervision.
    
    Args:
        map_grid_conf (dict): Map grid configuration with xbound, ybound, zbound
        thickness (int): Line thickness for rasterization. Default: 3
    """
    
    def __init__(self, map_grid_conf, thickness=3):
        self.map_grid_conf = map_grid_conf
        self.thickness = thickness
        
        # Calculate canvas size from grid config
        xbound = map_grid_conf['xbound']
        ybound = map_grid_conf['ybound']
        
        # Grid bounds: [min, max, resolution]
        self.patch_size = (ybound[1] - ybound[0], xbound[1] - xbound[0])
        
        # Canvas size (number of pixels)
        self.canvas_size = (
            int((ybound[1] - ybound[0]) / ybound[2]),
            int((xbound[1] - xbound[0]) / xbound[2])
        )
        
        # Scale factors for coordinate transformation
        self.scale_x = self.canvas_size[1] / self.patch_size[1]
        self.scale_y = self.canvas_size[0] / self.patch_size[0]
    
    def __call__(self, results):
        """
        Generate rasterized semantic indices from vector annotations.
        
        Args:
            results (dict): Result dict containing 'annotation' with vectorized maps
            
        Returns:
            dict: Updated results with 'semantic_indices' added
        """
        # Check if annotation exists
        if 'annotation' not in results and 'ann_info' not in results:
            # No annotation, create empty semantic mask
            results['semantic_indices'] = np.zeros(
                (1, self.canvas_size[0], self.canvas_size[1]), 
                dtype=np.uint8
            )
            return results
        
        # Get annotation
        annotation = results.get('annotation', results.get('ann_info', {}))
        
        # Create empty semantic mask
        # Shape: (num_classes, H, W)
        # For now, use single channel (binary mask)
        semantic_mask = np.zeros(
            (1, self.canvas_size[0], self.canvas_size[1]), 
            dtype=np.uint8
        )
        
        # Rasterize each vector class
        if isinstance(annotation, dict):
            for class_name, instances in annotation.items():
                if isinstance(instances, list):
                    for instance in instances:
                        if isinstance(instance, (list, np.ndarray)):
                            # Convert to LineString
                            if len(instance) >= 2:
                                line = LineString(instance)
                                self._rasterize_line(line, semantic_mask[0])
        
        # Store in results
        results['semantic_indices'] = semantic_mask
        
        return results
    
    def _rasterize_line(self, line, mask):
        """
        Rasterize a single line to the mask.
        
        Args:
            line (LineString): Line to rasterize
            mask (ndarray): Mask to draw on (H, W)
        """
        # Transform line to canvas coordinates
        trans_x = self.canvas_size[1] / 2
        trans_y = self.canvas_size[0] / 2
        
        # Scale and translate
        line_canvas = affinity.scale(
            line, 
            self.scale_x, 
            self.scale_y, 
            origin=(0, 0)
        )
        line_canvas = affinity.affine_transform(
            line_canvas,
            [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
        )
        
        # Get coordinates
        coords = np.array(list(line_canvas.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        
        if len(coords) >= 2:
            # Draw line on mask
            cv2.polylines(
                mask, 
                [coords], 
                False, 
                color=1, 
                thickness=self.thickness
            )
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(map_grid_conf={self.map_grid_conf}, '
        repr_str += f'thickness={self.thickness})'
        return repr_str