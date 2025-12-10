import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18
from ..modules.builder import SEG_ENCODER


# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
#                               align_corners=True)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x1 = torch.cat([x2, x1], dim=1)
#         return self.conv(x1)


# @SEG_ENCODER.register_module()
# class SegEncode(nn.Module):
#     """
#     SegEncode module for BEV segmentation.
    
#     Args:
#         inC (int): Input channels (e.g., 256)
#         outC (int): Output classes (e.g., 4)
#         size (tuple): Target output size (H, W), e.g., (200, 400)
#     """
#     def __init__(self, inC, outC, size):
#         super(SegEncode, self).__init__()
        
#         # Get ResNet18 trunk
#         trunk = resnet18(pretrained=False, zero_init_residual=True)
        
#         # Initial conv layer - adapt from inC to 64 channels
#         self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = trunk.bn1
#         self.relu = trunk.relu
        
#         # Upsample to target size
#         self.up_sampler = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        
#         # ResNet layers
#         self.layer1 = trunk.layer1  # 64 channels, stride 1
#         self.layer2 = trunk.layer2  # 128 channels, stride 2
#         self.layer3 = trunk.layer3  # 256 channels, stride 2
        
#         # Decoder
#         self.up1 = Up(64 + 256, 256, scale_factor=4)
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, outC, kernel_size=1, padding=0),
#         )
    
#     def forward(self, x):
#         """
#         Forward pass.
        
#         Args:
#             x: (B, C, H, W) - BEV features, e.g., (2, 256, 200, 100)
        
#         Returns:
#             seg: (B, outC, H_target, W_target) - Segmentation logits, e.g., (2, 4, 200, 400)
#         """
#         # Upsample to target size
#         x = self.up_sampler(x)  # (B, 256, 200, 400)
        
#         # Initial conv
#         x = self.conv1(x)  # (B, 64, 100, 200) - stride 2
#         x = self.bn1(x)
#         x = self.relu(x)
        
#         # Encoder
#         x1 = self.layer1(x)  # (B, 64, 100, 200)
#         x = self.layer2(x1)  # (B, 128, 50, 100) - stride 2
#         x2 = self.layer3(x)  # (B, 256, 25, 50) - stride 2
        
#         # Decoder
#         x = self.up1(x2, x1)  # (B, 256, 100, 200) - upsample 4x and concat
#         x = self.up2(x)  # (B, outC, 200, 400) - final upsample 2x and classify
        
#         return x
class Up(nn.Module):
    def __init__(self, inC, outC, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor,
                              mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_deep, x_skip):
        x_deep = self.up(x_deep)
        x = torch.cat([x_skip, x_deep], dim=1)
        return self.conv(x)

# @SEG_ENCODER.register_module()
# class SegEncode(nn.Module):
#     def __init__(self, inC, outC, size):
#         super().__init__()

#         self.target_size = size
#         self.resize = nn.Upsample(size=size, mode='bilinear', align_corners=True)

#         # --- stem: keep resolution, no stride 2 ---
#         self.stem = nn.Sequential(
#             nn.Conv2d(inC, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )

#         trunk = resnet18(pretrained=False)

#         self.layer1 = trunk.layer1      # 64 ch, same H,W
#         self.layer2 = trunk.layer2      # 128 ch, H/2,W/2
#         self.layer3 = trunk.layer3      # 256 ch, H/4,W/4
#         self.layer4 = trunk.layer4      # 512 ch, H/8,W/8

#         # Up(deep+skip, out, scale_factor)
#         self.up4 = Up(512 + 256, 256, scale_factor=2)  # x4 + x3
#         self.up3 = Up(256 + 128, 128, scale_factor=2)  # d3 + x2
#         self.up2 = Up(128 + 64,  64,  scale_factor=2)  # d2 + x1
#         self.up1 = Up(64  + 64,  64,  scale_factor=1)  # d1 + x0 (same res)

#         self.final = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, outC, kernel_size=1),
#         )

#     def forward(self, x):
#         # x: (B, inC, H, W)  e.g. BEV features
#         x0 = self.stem(x)               # (B, 64,   H,   W)
#         x1 = self.layer1(x0)            # (B, 64,   H,   W)
#         x2 = self.layer2(x1)            # (B,128,  H/2, W/2)
#         x3 = self.layer3(x2)            # (B,256,  H/4, W/4)
#         x4 = self.layer4(x3)            # (B,512,  H/8, W/8)

#         d3 = self.up4(x4, x3)           # (B,256,  H/4, W/4)
#         d2 = self.up3(d3, x2)           # (B,128,  H/2, W/2)
#         d1 = self.up2(d2, x1)           # (B, 64,   H,   W)
#         d0 = self.up1(d1, x0)           # (B, 64,   H,   W)

#         out = self.final(d0)            # (B, outC, H, W)
#         out = self.resize(out)          # (B, outC, H_target, W_target)

#         return out

@SEG_ENCODER.register_module()
class SegEncode(nn.Module):
    # def __init__(self, in_channels=256, num_classes=3, **kwargs):
    def __init__(self, inC, outC, size):
        super().__init__()
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(inC, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, outC, kernel_size=1)
        )
        self.resize = nn.Upsample(size=size, mode='bilinear', align_corners=True)

    def forward(self, bev_feat):
        """
        bev_feat: (B, C, H, W)
        returns segmentation logits (B, num_classes, H, W)
        """
        out = self.seg_head(bev_feat)
        out = self.resize(out)          # (B, outC, H_target, W_target)

        return out



@SEG_ENCODER.register_module()
class SegEncode_v1(nn.Module):

    def __init__(self, inC, outC):
        super(SegEncode_v1, self).__init__()
        self.seg_head = nn.Sequential(
            nn.Conv2d(inC, inC, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC , kernel_size=1))

    def forward(self, x):

        return self.seg_head(x)


import math
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch.nn.functional as F

@SEG_ENCODER.register_module()
class DeconvEncode(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels,
                 outC=4,
                 use_dcn=True,
                 init_cfg=None):
        super(DeconvEncode, self).__init__(init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
                                                     num_deconv_kernels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(num_deconv_filters[-1], num_deconv_filters[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_deconv_filters[-1], outC , kernel_size=1))

    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channel = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channel,
                feat_channel,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=dict(type='BN'))
            layers.append(conv_module)
            upsample_module = ConvModule(
                feat_channel,
                feat_channel,
                num_deconv_kernels[i],
                stride=2,
                padding=1,
                conv_cfg=dict(type='deconv'),
                norm_cfg=dict(type='BN'))
            layers.append(upsample_module)
            self.in_channel = feat_channel

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        outs = self.deconv_layers(inputs)
        outs = self.seg_head(outs)
        return outs

"""

dict(
type='DeconvEncode',
in_channel=256,
num_deconv_filters=(256, 128, 64),
num_deconv_kernels=(4, 4, 4),
use_dcn=True),
        
"""


@SEG_ENCODER.register_module()
class SegEncodeASPP(nn.Module):
    """
    ASPP-inspired head for capturing multi-scale context.
    """
    def __init__(self, inC, outC, size):
        super().__init__()
        self.target_size = size
        
        # Multi-scale parallel branches
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inC, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3x3_d1 = nn.Sequential(
            nn.Conv2d(inC, 64, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3x3_d2 = nn.Sequential(
            nn.Conv2d(inC, 64, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3x3_d4 = nn.Sequential(
            nn.Conv2d(inC, 64, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Global context - NO BatchNorm after 1x1 pooling!
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inC, 64, 1, bias=True),  # bias=True since no BN
            nn.ReLU(inplace=True)
        )
        
        # Fusion: 5 branches × 64 = 320 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 5, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        # Progressive upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Conv2d(32, outC, 1)
        self.final_resize = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # x: (B, 256, 50, 50)
        size = x.shape[-2:]
        
        # Multi-scale features
        f1 = self.conv1x1(x)
        f2 = self.conv3x3_d1(x)
        f3 = self.conv3x3_d2(x)
        f4 = self.conv3x3_d4(x)
        f5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True)
        
        # Concatenate and fuse
        x = torch.cat([f1, f2, f3, f4, f5], dim=1)  # (B, 320, 50, 50)
        x = self.fusion(x)     # (B, 128, 50, 50)
        x = self.upsample(x)   # (B, 32, 200, 200)
        x = self.classifier(x) # (B, 3, 200, 200)
        x = self.final_resize(x)
        return x

@SEG_ENCODER.register_module()
class SegEncodeV2(nn.Module):
    """
    Improved segmentation head with:
    - Progressive upsampling (learn the upsampling, don't just interpolate)
    - Residual connections for gradient flow
    - Separate spatial and channel processing
    """
    def __init__(self, inC, outC, size):
        super().__init__()
        self.target_size = size  # (200, 400)
        
        # Stage 1: Process at input resolution (50x50)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inC, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Stage 2: Upsample 2x (50x50 → 100x100) with learned conv
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Stage 3: Upsample 2x (100x100 → 200x200) with learned conv
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(32, outC, kernel_size=1)
        
        # Final resize to exact target (handles non-power-of-2)
        self.final_resize = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # x: (B, 256, 50, 50)
        x = self.conv1(x)      # (B, 128, 50, 50)
        x = self.up1(x)        # (B, 64, 100, 100)
        x = self.up2(x)        # (B, 32, 200, 200)
        x = self.classifier(x) # (B, 3, 200, 200)
        x = self.final_resize(x)  # (B, 3, 200, 400)
        return x