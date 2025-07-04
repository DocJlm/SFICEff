# network/MainNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入原始模块
try:
    from . import SFIConvResnet
except ImportError:
    import SFIConvResnet

# 导入增强模块
try:
    from .enhanced_modules_v2 import AMSFEModule, CLFPFModule, SALGAModule
except ImportError:
    try:
        from enhanced_modules_v2 import AMSFEModule, CLFPFModule, SALGAModule
    except ImportError:
        print("Warning: Enhanced modules not found, using placeholder modules")
        
        # 占位符模块
        class AMSFEModule(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.identity = nn.Identity()
            def forward(self, x):
                return self.identity(x)
        
        class CLFPFModule(nn.Module):
            def __init__(self, in_channels, out_channels=256):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, 1)
            def forward(self, x):
                return self.conv(x)
        
        class SALGAModule(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.identity = nn.Identity()
            def forward(self, x):
                return self.identity(x)

class MainNet(nn.Module):
    """原始MainNet"""
    def __init__(self, num_classes=2):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        
        # 创建SFI-ResNet骨干
        self.backbone = SFIConvResnet.SFIresnet26(pretrained=False)
        
        # 提取特征部分（去掉最后的分类层）
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
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        
        # 处理SFI双分支输出
        if isinstance(features, tuple):
            # 如果是(spatial, frequency)元组，取spatial分支
            features = features[0]
        
        # 分类
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.fc(features)
        
        return output

class EnhancedMainNet(nn.Module):
    """增强版MainNet"""
    def __init__(self, num_classes=2, enable_amsfe=True, enable_clfpf=True, enable_salga=True):
        super(EnhancedMainNet, self).__init__()
        self.num_classes = num_classes
        self.enable_amsfe = enable_amsfe
        self.enable_clfpf = enable_clfpf
        self.enable_salga = enable_salga
        
        # 骨干网络
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
        
        # 基础特征通道数
        base_channels = 2048
        current_channels = base_channels
        
        # 创新点模块
        if enable_amsfe:
            self.amsfe = AMSFEModule(current_channels)
            print(f"AMSFE enabled with {current_channels} channels")
        
        if enable_clfpf:
            self.clfpf = CLFPFModule(current_channels, 256)
            current_channels = 256
            print(f"CLFPF enabled, output channels: {current_channels}")
        
        if enable_salga:
            self.salga = SALGAModule(current_channels)
            print(f"SALGA enabled with {current_channels} channels")
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(current_channels, num_classes)
        
        print(f"Enhanced MainNet initialized with {current_channels} -> {num_classes}")
        
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        
        # 处理SFI双分支输出
        if isinstance(features, tuple):
            features = features[0]  # 取spatial分支
        
        # 应用创新点模块
        if self.enable_amsfe:
            features = self.amsfe(features)
        
        if self.enable_clfpf:
            features = self.clfpf(features)
        
        if self.enable_salga:
            features = self.salga(features)
        
        # 分类
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        
        return output


