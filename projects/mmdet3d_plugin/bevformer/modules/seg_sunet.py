import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from .builder import SEG_ENCODER

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


@SEG_ENCODER.register_module()
class SegEncode(nn.Module):
    """
    SegEncode module for BEV segmentation.
    
    Args:
        inC (int): Input channels (e.g., 256)
        outC (int): Output classes (e.g., 4)
        size (tuple): Target output size (H, W), e.g., (200, 400)
    """
    def __init__(self, inC, outC, size):
        super(SegEncode, self).__init__()
        
        # Get ResNet18 trunk
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        
        # Initial conv layer - adapt from inC to 64 channels
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        
        # Upsample to target size
        self.up_sampler = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        
        # ResNet layers
        self.layer1 = trunk.layer1  # 64 channels, stride 1
        self.layer2 = trunk.layer2  # 128 channels, stride 2
        self.layer3 = trunk.layer3  # 256 channels, stride 2
        
        # Decoder
        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) - BEV features, e.g., (2, 256, 200, 100)
        
        Returns:
            seg: (B, outC, H_target, W_target) - Segmentation logits, e.g., (2, 4, 200, 400)
        """
        # Upsample to target size
        x = self.up_sampler(x)  # (B, 256, 200, 400)
        
        # Initial conv
        x = self.conv1(x)  # (B, 64, 100, 200) - stride 2
        x = self.bn1(x)
        x = self.relu(x)
        
        # Encoder
        x1 = self.layer1(x)  # (B, 64, 100, 200)
        x = self.layer2(x1)  # (B, 128, 50, 100) - stride 2
        x2 = self.layer3(x)  # (B, 256, 25, 50) - stride 2
        
        # Decoder
        x = self.up1(x2, x1)  # (B, 256, 100, 200) - upsample 4x and concat
        x = self.up2(x)  # (B, outC, 200, 400) - final upsample 2x and classify
        
        return x