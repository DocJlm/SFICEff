# network/enhanced_modules_v2_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdvancedAMSFEModule(nn.Module):
    """高级自适应多尺度频域增强模块"""
    def __init__(self, channels, reduction=16):
        super(AdvancedAMSFEModule, self).__init__()
        
        # 多尺度频域特征提取 - 修复通道维度
        self.freq_branches = nn.ModuleList([
            # 低频分支
            nn.Sequential(
                nn.Conv2d(channels, channels//4, 1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels//4, channels//4, 3, padding=1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True)
            ),
            # 中频分支
            nn.Sequential(
                nn.Conv2d(channels, channels//4, 1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels//4, channels//4, 3, padding=2, dilation=2),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True)
            ),
            # 高频分支
            nn.Sequential(
                nn.Conv2d(channels, channels//4, 1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels//4, channels//4, 3, padding=4, dilation=4),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True)
            ),
            # 超高频分支
            nn.Sequential(
                nn.Conv2d(channels, channels//4, 1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels//4, channels//4, 5, padding=2),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 自适应权重生成
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, 4, 1),  # 4个分支
            nn.Softmax(dim=1)
        )
        
        # 特征融合 - 确保输入输出通道一致
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),  # 4*(channels//4) = channels
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # 获取自适应权重
        weights = self.weight_generator(x)  # B x 4 x 1 x 1
        
        # 多尺度特征提取
        branch_features = []
        for i, branch in enumerate(self.freq_branches):
            feat = branch(x)
            weighted_feat = feat * weights[:, i:i+1, :, :]
            branch_features.append(weighted_feat)
        
        # 特征拼接
        multi_scale = torch.cat(branch_features, dim=1)  # B x channels x H x W
        
        # 特征融合
        enhanced = self.fusion(multi_scale)
        
        # 自适应残差连接
        output = enhanced + self.residual_weight * x
        
        return output

class FixedCLFPFModule(nn.Module):
    """修复的跨层特征金字塔融合模块"""
    def __init__(self, in_channels, out_channels=256):
        super(FixedCLFPFModule, self).__init__()
        
        # 通道对齐
        self.channel_align = nn.Conv2d(in_channels, out_channels, 1)
        
        # 计算各层通道数，确保可以正确相加
        mid_channels = out_channels // 2
        quarter_channels = out_channels // 4
        
        # 分层特征提取 - 修复通道维度
        self.level1 = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.level2 = nn.Sequential(
            nn.Conv2d(mid_channels, quarter_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(quarter_channels),
            nn.ReLU(inplace=True)
        )
        
        self.level3 = nn.Sequential(
            nn.Conv2d(quarter_channels, quarter_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(quarter_channels),
            nn.ReLU(inplace=True)
        )
        
        # 自底向上路径 - 确保通道匹配
        self.upward_conv2 = nn.Sequential(
            nn.Conv2d(quarter_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.upward_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//16, out_channels, 1),
            nn.Sigmoid()
        )
        
        print(f"CLFPF initialized: {in_channels} -> {out_channels}")
        print(f"Level channels: {out_channels} -> {mid_channels} -> {quarter_channels}")
        
    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)  # B x out_channels x H x W
        
        # 自顶向下
        l1 = self.level1(x)        # B x mid_channels x H x W
        l2 = self.level2(l1)       # B x quarter_channels x H x W  
        l3 = self.level3(l2)       # B x quarter_channels x H x W
        
        # 自底向上 - 现在通道匹配了
        up2 = self.upward_conv2(l3) + l1  # both are mid_channels
        up1 = self.upward_conv1(up2) + x  # both are out_channels
        
        # 注意力加权
        att = self.attention(up1)
        output = up1 * att
        
        return output

class SimplifiedSALGAModule(nn.Module):
    """简化的语义感知局部-全局注意力模块"""
    def __init__(self, channels):
        super(SimplifiedSALGAModule, self).__init__()
        self.channels = channels
        
        # 局部注意力分支
        self.local_attention = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.BatchNorm2d(channels//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, 7, padding=3, groups=channels//8),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # 全局注意力分支
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//16, channels, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 局部和全局注意力
        local_att = self.local_attention(x)
        global_att = self.global_attention(x)
        channel_att = self.channel_attention(x)
        
        # 特征增强
        local_feat = x * local_att
        global_feat = x * global_att
        channel_feat = x * channel_att
        
        # 组合特征
        combined = local_feat + global_feat + channel_feat
        
        # 融合和残差连接
        output = self.fusion(combined) + x
        
        return output

class SuperEnhancedMainNet(nn.Module):
    """超级增强版MainNet - 修复版"""
    def __init__(self, num_classes=2):
        super(SuperEnhancedMainNet, self).__init__()
        
        # 导入SFI骨干
        try:
            from . import SFIConvResnet
        except ImportError:
            import SFIConvResnet
            
        self.backbone = SFIConvResnet.SFIresnet26(pretrained=False)
        
        # 特征提取器
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )
        
        # 增强模块 - 修复通道匹配
        base_channels = 2048
        intermediate_channels = 512
        
        print(f"Initializing modules with {base_channels} base channels")
        
        self.amsfe = AdvancedAMSFEModule(base_channels)
        self.clfpf = FixedCLFPFModule(base_channels, intermediate_channels)
        self.salga = SimplifiedSALGAModule(intermediate_channels)
        
        # 特征压缩和分类
        self.feature_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(intermediate_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"SuperEnhancedMainNet initialized: {base_channels} -> {intermediate_channels} -> {num_classes}")
        
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        
        # 处理SFI双分支输出
        if isinstance(features, tuple):
            features = features[0]  # 取spatial分支
        
        # 打印调试信息
        # print(f"Backbone output shape: {features.shape}")
        
        # 应用增强模块
        features = self.amsfe(features)
        # print(f"After AMSFE: {features.shape}")
        
        features = self.clfpf(features)
        # print(f"After CLFPF: {features.shape}")
        
        features = self.salga(features)
        # print(f"After SALGA: {features.shape}")
        
        # 分类
        output = self.feature_compress(features)
        
        return output

# 损失函数保持不变
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, focal_weight=0.7, ce_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.focal_weight * focal + self.ce_weight * ce