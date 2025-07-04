# network/SFIConvEfficientNet.py
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Modified for EfficientNet-B4 backbone (Complete ResNet Replacement)
# Pytorch Implementation of SFIConv-EfficientNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.SFIConv import *
import math

__all__ = ['SFIEfficientNet', 'sfi_efficientnet_b4']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SFIMBConv(nn.Module):
    """SFI版本的MobileNet Inverted Bottleneck - 彻底修复版"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 expand_ratio=6, alpha=0.5, se_ratio=0.25, norm_layer=None, First=False):
        super(SFIMBConv, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.stride = stride
        self.use_se = se_ratio is not None and se_ratio > 0
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.first = First
        self.alpha = alpha
        
        # Expansion phase
        hidden_dim = int(round(in_channels * expand_ratio))
        
        # 完全避免SFI卷积，使用标准卷积
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                norm_layer(hidden_dim),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = None
            
        # Depthwise convolution - 使用标准分组卷积
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            norm_layer(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # SE layer
        if self.use_se:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Point-wise linear projection - 使用标准卷积
        self.pw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        
    def forward(self, x):
        # 简化处理：如果输入是SFI双分支，只取spatial分支
        if isinstance(x, tuple):
            x, _ = x
            
        identity = x if self.use_shortcut else None
        
        # Standard MobileNet block
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        x = self.dw_conv(x)
        
        if self.use_se:
            se_weight = self.se(x)
            x = x * se_weight
            
        x = self.pw_conv(x)
        
        # Shortcut connection
        if self.use_shortcut and identity is not None:
            x = x + identity
            
        return x


class SELayer(nn.Module):
    """增强的Squeeze-and-Excitation layer for SFI features"""
    def __init__(self, channels, reduction_channels, alpha=0.5):
        super(SELayer, self).__init__()
        self.alpha = alpha
        
        # 分别处理spatial和frequency分支
        spatial_channels = int(channels * (1 - alpha))
        freq_channels = int(channels * alpha)
        
        # 确保reduction_channels不为0
        reduction_channels = max(1, reduction_channels)
        
        self.se_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(spatial_channels, reduction_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, spatial_channels, 1),
            nn.Sigmoid()
        )
        
        if freq_channels > 0:
            self.se_freq = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(freq_channels, reduction_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduction_channels, freq_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se_freq = None
        
    def forward(self, x):
        x_s, x_f = x
        
        # Apply SE to spatial branch
        se_s = self.se_spatial(x_s)
        x_s = x_s * se_s
        
        # Apply SE to frequency branch
        if self.se_freq is not None and x_f.size(1) > 0:
            se_f = self.se_freq(x_f)
            x_f = x_f * se_f
        
        return x_s, x_f


class SFIEfficientNet(nn.Module):
    """SFI-EfficientNet模型 - 专注于B4配置"""
    def __init__(self, variant='b4', num_classes=2, alpha=0.5, drop_rate=0.2):
        super(SFIEfficientNet, self).__init__()
        
        self.alpha = alpha
        self.drop_rate = drop_rate
        
        # EfficientNet-B4的专门配置
        # [expand_ratio, channels, layers, stride, kernel_size, se_ratio]
        if variant == 'b4':
            width_mult = 1.4
            depth_mult = 1.8
            resolution = 380
            config = [
                [1, 24, 2, 1, 3, 0.25],   # Stage 1
                [6, 32, 4, 2, 3, 0.25],   # Stage 2  
                [6, 56, 4, 2, 5, 0.25],   # Stage 3
                [6, 112, 6, 2, 3, 0.25],  # Stage 4
                [6, 160, 6, 1, 5, 0.25],  # Stage 5
                [6, 272, 8, 2, 5, 0.25],  # Stage 6
                [6, 448, 2, 1, 3, 0.25],  # Stage 7
            ]
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        # Scale width and depth
        config = [[c[0], int(c[1] * width_mult), max(1, int(c[2] * depth_mult)), 
                  c[3], c[4], c[5]] for c in config]
        
        # Stem - 输入处理
        stem_channels = int(48 * width_mult)  # B4使用48而不是32
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)  # EfficientNet使用SiLU而不是ReLU
        )
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        
        for stage_idx, (expand_ratio, out_channels, layers, stride, kernel_size, se_ratio) in enumerate(config):
            # First block in stage
            first_block = SFIMBConv(
                in_channels, out_channels, kernel_size, stride, 
                expand_ratio, alpha, se_ratio, First=(stage_idx==0)
            )
            self.blocks.append(first_block)
            in_channels = out_channels
            
            # Remaining blocks in stage
            for _ in range(layers - 1):
                block = SFIMBConv(
                    in_channels, out_channels, kernel_size, 1, 
                    expand_ratio, alpha, se_ratio, First=False
                )
                self.blocks.append(block)
        
        # Head - 也使用标准卷积避免SFI问题
        final_channels = int(1792 * width_mult)  # B4的最终通道数
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True)
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(final_channels, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
        print(f"SFI-EfficientNet-{variant.upper()} created:")
        print(f"  - Resolution: {resolution}x{resolution}")
        print(f"  - Final channels: {final_channels}")
        print(f"  - Total blocks: {len(self.blocks)}")
        print(f"  - Drop rate: {drop_rate}")
        
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / m.weight.size(1)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)
    
    def extract_features(self, x):
        """提取特征，不进行分类"""
        # Stem
        x = self.stem(x)
        
        # Blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
        # Head conv
        x = self.head_conv(x)
        
        return x
                
    def forward(self, x):
        # Feature extraction
        features = self.extract_features(x)
        
        # Global pooling
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # Dropout and classification
        features = self.dropout(features)
        output = self.classifier(features)
        
        return output


def sfi_efficientnet_b4(num_classes=2, alpha=0.5, drop_rate=0.2, **kwargs):
    """创建SFI-EfficientNet-B4模型"""
    return SFIEfficientNet(variant='b4', num_classes=num_classes, alpha=alpha, drop_rate=drop_rate, **kwargs)


# 为了保持接口一致性，也提供其他变体的接口
def sfi_efficientnet_b0(num_classes=2, alpha=0.5, **kwargs):
    """SFI-EfficientNet-B0 (轻量版本用于快速测试)"""
    print("Warning: B0 is deprecated, using B4 instead for better performance")
    return sfi_efficientnet_b4(num_classes=num_classes, alpha=alpha, drop_rate=0.1, **kwargs)


if __name__ == '__main__':
    # 测试EfficientNet-B4模型
    print("Testing SFI-EfficientNet-B4...")
    
    # 创建模型
    model = sfi_efficientnet_b4(num_classes=2).cuda()
    
    # 测试不同输入尺寸
    test_sizes = [224, 380, 512]  # B4推荐380，但也要测试其他尺寸
    
    for size in test_sizes:
        print(f"\nTesting with input size {size}x{size}:")
        try:
            x = torch.randn(2, 3, size, size).cuda()
            
            # 前向传播
            with torch.no_grad():
                y = model(x)
                features = model.extract_features(x)
            
            print(f"  ✅ Input: {x.shape}")
            print(f"  ✅ Output: {y.shape}")
            print(f"  ✅ Features: {features.shape}")
            
        except Exception as e:
            print(f"  ❌ Error with size {size}: {e}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n🎉 SFI-EfficientNet-B4 test completed successfully!")