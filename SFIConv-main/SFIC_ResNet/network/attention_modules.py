import torch
import torch.nn as nn
import torch.nn.functional as F

class SALGAModule(nn.Module):
    """语义感知的局部-全局注意力模块"""
    def __init__(self, channels, num_semantic_regions=5, local_window_size=7):
        super(SALGAModule, self).__init__()
        self.channels = channels
        self.num_semantic_regions = num_semantic_regions
        self.local_window_size = local_window_size
        
        # 语义区域分割器（简化版，基于特征聚类）
        self.semantic_segmenter = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, num_semantic_regions, 1),
            nn.Softmax(dim=1)  # 生成软分割图
        )
        
        # 全局注意力分支
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        # 局部注意力分支
        self.local_attention = LocalAttentionBlock(channels, local_window_size)
        
        # 语义感知的特征增强器
        self.semantic_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_semantic_regions)
        ])
        
        # 注意力融合
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),  # global + local + semantic
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 语义区域分割
        semantic_maps = self.semantic_segmenter(x)  # (B, num_regions, H, W)
        
        # 全局注意力
        global_att = self.global_attention(x)  # (B, C, 1, 1)
        global_enhanced = x * global_att
        
        # 局部注意力
        local_enhanced = self.local_attention(x)
        
        # 语义感知增强
        semantic_enhanced_features = []
        for i in range(self.num_semantic_regions):
            region_mask = semantic_maps[:, i:i+1, :, :]  # (B, 1, H, W)
            region_feature = x * region_mask
            enhanced = self.semantic_enhancers[i](region_feature)
            semantic_enhanced_features.append(enhanced * region_mask)
        
        semantic_enhanced = sum(semantic_enhanced_features)
        
        # 融合三种注意力
        combined = torch.cat([global_enhanced, local_enhanced, semantic_enhanced], dim=1)
        output = self.attention_fusion(combined)
        
        return output

class LocalAttentionBlock(nn.Module):
    """局部注意力块"""
    def __init__(self, channels, window_size=7):
        super(LocalAttentionBlock, self).__init__()
        self.window_size = window_size
        self.channels = channels
        
        # 局部特征提取
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, window_size, padding=window_size//2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        local_att = self.local_conv(x)
        return x * local_att