# network/EfficientNetB4Enhanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from efficientnet_pytorch import EfficientNet

class AdvancedAMSFEModule(nn.Module):
    """高级自适应多尺度频域增强模块"""
    def __init__(self, channels, reduction=16):
        super(AdvancedAMSFEModule, self).__init__()
        
        # 多尺度频域特征提取
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
            nn.Conv2d(channels//reduction, 4, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接权重
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
        
        # 特征拼接和融合
        multi_scale = torch.cat(branch_features, dim=1)
        enhanced = self.fusion(multi_scale)
        
        # 残差连接
        output = enhanced + self.residual_weight * x
        
        return output


class FixedCLFPFModule(nn.Module):
    """跨层特征金字塔融合模块"""
    def __init__(self, in_channels, out_channels=256):
        super(FixedCLFPFModule, self).__init__()
        
        # 通道对齐
        self.channel_align = nn.Conv2d(in_channels, out_channels, 1)
        
        # 计算各层通道数
        mid_channels = out_channels // 2
        quarter_channels = out_channels // 4
        
        # 分层特征提取
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
        
        # 自底向上路径
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
        
    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)
        
        # 自顶向下
        l1 = self.level1(x)
        l2 = self.level2(l1)
        l3 = self.level3(l2)
        
        # 自底向上
        up2 = self.upward_conv2(l3) + l1
        up1 = self.upward_conv1(up2) + x
        
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
        # 三种注意力
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


class EfficientNetB4Enhanced(nn.Module):
    """EfficientNet-B4 + 三个创新点"""
    def __init__(self, num_classes=2, drop_rate=0.3, 
                 enable_amsfe=True, enable_clfpf=True, enable_salga=True):
        super(EfficientNetB4Enhanced, self).__init__()
        
        self.num_classes = num_classes
        self.enable_amsfe = enable_amsfe
        self.enable_clfpf = enable_clfpf
        self.enable_salga = enable_salga
        
        # 使用预训练的EfficientNet-B4作为骨干网络
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        
        # 获取backbone的特征提取部分（去掉分类器）
        self.features = self.backbone.extract_features
        
        # 动态获取特征通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_features = self.features(test_input)
            feature_channels = test_features.shape[1]
        
        print(f"EfficientNet-B4 feature channels: {feature_channels}")
        
        current_channels = feature_channels
        
        # 添加三个创新模块
        if enable_amsfe:
            self.amsfe = AdvancedAMSFEModule(current_channels)
            print(f"AMSFE enabled with {current_channels} channels")
        
        if enable_clfpf:
            self.clfpf = FixedCLFPFModule(current_channels, 512)
            current_channels = 512
            print(f"CLFPF enabled, output channels: {current_channels}")
        
        if enable_salga:
            self.salga = SimplifiedSALGAModule(current_channels)
            print(f"SALGA enabled with {current_channels} channels")
        
        # 分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        
        # 更强的分类器
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, current_channels // 2),
            nn.BatchNorm1d(current_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(current_channels // 2, num_classes)
        )
        
        print(f"EfficientNet-B4 Enhanced initialized:")
        print(f"  - Backbone: EfficientNet-B4 (pretrained)")
        print(f"  - Feature flow: {feature_channels} -> {current_channels} -> {num_classes}")
        print(f"  - Enhanced modules: AMSFE={enable_amsfe}, CLFPF={enable_clfpf}, SALGA={enable_salga}")
        
    def extract_features(self, x):
        """提取骨干网络特征"""
        return self.features(x)
        
    def forward(self, x):
        # EfficientNet特征提取
        features = self.features(x)
        
        # 应用增强模块
        if self.enable_amsfe:
            features = self.amsfe(features)
        
        if self.enable_clfpf:
            features = self.clfpf(features)
        
        if self.enable_salga:
            features = self.salga(features)
        
        # 全局池化和分类
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        output = self.classifier(features)
        
        return output


# 创建模型的函数
def create_enhanced_efficientnet_b4(num_classes=2, drop_rate=0.3, 
                                   enable_amsfe=True, enable_clfpf=True, enable_salga=True):
    """创建增强版EfficientNet-B4"""
    return EfficientNetB4Enhanced(
        num_classes=num_classes,
        drop_rate=drop_rate,
        enable_amsfe=enable_amsfe,
        enable_clfpf=enable_clfpf,
        enable_salga=enable_salga
    )


if __name__ == '__main__':
    print("🚀 Testing EfficientNet-B4 Enhanced")
    
    # 测试不同配置
    configs = [
        ('All modules', True, True, True),
        ('Without SALGA', True, True, False),
        ('Only AMSFE', True, False, False),
        ('Baseline', False, False, False)
    ]
    
    for name, amsfe, clfpf, salga in configs:
        print(f"\n{name}:")
        try:
            model = create_enhanced_efficientnet_b4(
                enable_amsfe=amsfe,
                enable_clfpf=clfpf,
                enable_salga=salga
            )
            
            # 测试前向传播
            x = torch.randn(2, 3, 224, 224)
            y = model(x)
            
            # 统计参数
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"  ✅ Output shape: {y.shape}")
            print(f"  📊 Parameters: {total_params:,}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n🎉 All tests completed!")