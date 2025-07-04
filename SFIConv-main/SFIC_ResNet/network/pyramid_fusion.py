import torch
import torch.nn as nn
import torch.nn.functional as F

class CLFPFModule(nn.Module):
    """跨层特征金字塔融合模块"""
    def __init__(self, feature_channels=[256, 512, 1024, 2048], 
                 unified_channels=256, alpha=0.5):
        super(CLFPFModule, self).__init__()
        self.feature_channels = feature_channels
        self.unified_channels = unified_channels
        self.alpha = alpha
        
        # 特征通道统一化
        self.channel_unifiers = nn.ModuleList()
        for channels in feature_channels:
            # 分别处理spatial和frequency分支
            spatial_channels = int(channels * (1 - alpha)) if channels != feature_channels[0] else channels
            freq_channels = int(channels * alpha) if channels != feature_channels[0] else 0
            
            if freq_channels > 0:
                self.channel_unifiers.append(nn.ModuleDict({
                    'spatial': nn.Conv2d(spatial_channels, unified_channels//2, 1),
                    'frequency': nn.Conv2d(freq_channels, unified_channels//2, 1)
                }))
            else:
                # 第一层没有frequency分支
                self.channel_unifiers.append(nn.ModuleDict({
                    'spatial': nn.Conv2d(spatial_channels, unified_channels, 1),
                    'frequency': None
                }))
        
        # 跨层注意力机制
        self.cross_layer_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(unified_channels, unified_channels//4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(unified_channels//4, unified_channels, 1),
                nn.Sigmoid()
            ) for _ in range(len(feature_channels))
        ])
        
        # 特征金字塔融合
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(unified_channels, unified_channels, 3, padding=1),
                nn.BatchNorm2d(unified_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(feature_channels))
        ])
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(unified_channels * len(feature_channels), unified_channels, 1),
            nn.BatchNorm2d(unified_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_channels, unified_channels, 3, padding=1),
            nn.BatchNorm2d(unified_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, multi_scale_features):
        """
        multi_scale_features: list of [(x_s, x_f), ...] 或 [x, ...] for first layer
        """
        unified_features = []
        
        # 统一化各层特征
        for i, (features, unifier) in enumerate(zip(multi_scale_features, self.channel_unifiers)):
            if isinstance(features, tuple):
                # SFI双分支特征
                x_s, x_f = features
                unified_s = unifier['spatial'](x_s)
                unified_f = unifier['frequency'](x_f)
                unified = torch.cat([unified_s, unified_f], dim=1)
            else:
                # 单分支特征（第一层）
                unified = unifier['spatial'](features)
            
            unified_features.append(unified)
        
        # 特征尺寸对齐到最大尺寸
        target_size = unified_features[0].shape[2:]  # 使用第一层的尺寸作为目标
        aligned_features = []
        
        for i, feat in enumerate(unified_features):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 应用跨层注意力
        attended_features = []
        for i, (feat, attention) in enumerate(zip(aligned_features, self.cross_layer_attention)):
            att_weight = attention(feat)
            attended_feat = feat * att_weight
            attended_features.append(attended_feat)
        
        # FPN式融合
        fpn_features = []
        for i, (feat, fpn_conv) in enumerate(zip(attended_features, self.fpn_convs)):
            fpn_feat = fpn_conv(feat)
            fpn_features.append(fpn_feat)
        
        # 最终融合
        concatenated = torch.cat(fpn_features, dim=1)
        fused_feature = self.final_fusion(concatenated)
        
        return fused_feature