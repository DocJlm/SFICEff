# network/MainNet.py - 更新版本，支持EfficientNet-B4 + 三个创新点
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入增强模块
try:
    from .enhanced_modules_v2 import AdvancedAMSFEModule, FixedCLFPFModule, SimplifiedSALGAModule
except ImportError:
    try:
        from enhanced_modules_v2 import AdvancedAMSFEModule, FixedCLFPFModule, SimplifiedSALGAModule
    except ImportError:
        print("Warning: Enhanced modules not found, using EfficientNet-B4 Enhanced")
        
        # 直接导入EfficientNet-B4增强版
        try:
            from .EfficientNetB4Enhanced import EfficientNetB4Enhanced
        except ImportError:
            from EfficientNetB4Enhanced import EfficientNetB4Enhanced

# 尝试导入EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    print("Warning: efficientnet_pytorch not installed. Please install with: pip install efficientnet_pytorch")
    EFFICIENTNET_AVAILABLE = False


class MainNet(nn.Module):
    """主网络 - 使用EfficientNet-B4作为骨干网络"""
    def __init__(self, num_classes=2, drop_rate=0.3):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        
        if not EFFICIENTNET_AVAILABLE:
            raise ImportError("EfficientNet not available. Please install efficientnet_pytorch")
        
        # 使用预训练的EfficientNet-B4
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # 移除原始分类器
        self.backbone._fc = nn.Identity()
        
        # 获取特征维度
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(test_input)
            feature_dim = features.shape[1]
        
        # 新的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        print(f"MainNet initialized with EfficientNet-B4")
        print(f"  - Classes: {num_classes}")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - Drop rate: {drop_rate}")
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """提取特征"""
        return self.backbone(x)


class EnhancedMainNet(nn.Module):
    """增强版MainNet - EfficientNet-B4 + 三个创新点"""
    def __init__(self, num_classes=2, drop_rate=0.3,
                 enable_amsfe=True, enable_clfpf=True, enable_salga=True):
        super(EnhancedMainNet, self).__init__()
        
        self.num_classes = num_classes
        self.enable_amsfe = enable_amsfe
        self.enable_clfpf = enable_clfpf
        self.enable_salga = enable_salga
        
        if not EFFICIENTNET_AVAILABLE:
            raise ImportError("EfficientNet not available. Please install efficientnet_pytorch")
        
        # 使用预训练的EfficientNet-B4作为特征提取器
        backbone = EfficientNet.from_pretrained('efficientnet-b4')
        
        # 提取特征部分（去掉分类器和最后的池化层）
        self.feature_extractor = nn.Sequential(
            backbone._conv_stem,
            backbone._bn0,
            backbone._swish,
            *backbone._blocks,
            backbone._conv_head,
            backbone._bn1,
            backbone._swish
        )
        
        # 动态获取特征通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_features = self.feature_extractor(test_input)
            self.feature_channels = test_features.shape[1]
        
        print(f"EfficientNet-B4 backbone feature channels: {self.feature_channels}")
        
        current_channels = self.feature_channels
        
        # 添加增强模块
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
        
        # 强化的分类器
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, current_channels // 2),
            nn.BatchNorm1d(current_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(current_channels // 2, num_classes)
        )
        
        print(f"Enhanced MainNet initialized:")
        print(f"  - Backbone: EfficientNet-B4 (pretrained)")
        print(f"  - Feature flow: {self.feature_channels} -> {current_channels} -> {num_classes}")
        print(f"  - Enhanced modules: AMSFE={enable_amsfe}, CLFPF={enable_clfpf}, SALGA={enable_salga}")
        
    def extract_features(self, x):
        """提取骨干网络特征"""
        return self.feature_extractor(x)
        
    def forward(self, x):
        # 特征提取
        features = self.extract_features(x)
        
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


class SuperEnhancedMainNet(nn.Module):
    """超级增强版MainNet"""
    def __init__(self, num_classes=2, drop_rate=0.3):
        super(SuperEnhancedMainNet, self).__init__()
        
        # 使用增强版，并启用所有模块
        self.enhanced_net = EnhancedMainNet(
            num_classes=num_classes,
            drop_rate=drop_rate,
            enable_amsfe=True,
            enable_clfpf=True,
            enable_salga=True
        )
        
        print("SuperEnhancedMainNet: All enhancement modules enabled")
        
    def forward(self, x):
        return self.enhanced_net(x)
    
    def extract_features(self, x):
        return self.enhanced_net.extract_features(x)


# 统一的模型创建函数
def create_model(model_type='standard', num_classes=2, drop_rate=0.3, **kwargs):
    """
    创建模型的统一接口
    
    Args:
        model_type: 'standard', 'enhanced', 'super' 
        num_classes: 分类数量
        drop_rate: Dropout比率
        **kwargs: 其他参数
    """
    if model_type == 'standard':
        return MainNet(num_classes=num_classes, drop_rate=drop_rate)
    elif model_type == 'enhanced':
        return EnhancedMainNet(num_classes=num_classes, drop_rate=drop_rate, **kwargs)
    elif model_type == 'super':
        return SuperEnhancedMainNet(num_classes=num_classes, drop_rate=drop_rate)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def model_complexity_analysis(model, input_size=(3, 224, 224)):
    """分析模型复杂度"""
    try:
        from thop import profile, clever_format
        
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        
        return {
            'flops': flops,
            'params': params,
            'flops_str': flops_str,
            'params_str': params_str
        }
    except ImportError:
        print("thop not available, skipping complexity analysis")
        return None


if __name__ == '__main__':
    print("🚀 Testing EfficientNet-B4 MainNet Models\n")
    
    if not EFFICIENTNET_AVAILABLE:
        print("❌ EfficientNet not available. Please install with:")
        print("pip install efficientnet_pytorch")
        exit(1)
    
    # 测试不同模型类型
    models_to_test = [
        ('standard', {}),
        ('enhanced', {'enable_amsfe': True, 'enable_clfpf': False, 'enable_salga': False}),
        ('enhanced', {'enable_amsfe': True, 'enable_clfpf': True, 'enable_salga': False}),
        ('enhanced', {'enable_amsfe': True, 'enable_clfpf': True, 'enable_salga': True}),
        ('super', {})
    ]
    
    input_size = (3, 224, 224)  # 标准输入尺寸
    batch_size = 2
    
    for model_type, kwargs in models_to_test:
        print(f"Testing {model_type} model with {kwargs}...")
        
        try:
            # 创建模型
            model = create_model(model_type=model_type, **kwargs)
            model = model.cuda() if torch.cuda.is_available() else model
            model.eval()
            
            # 测试前向传播
            x = torch.randn(batch_size, *input_size)
            if torch.cuda.is_available():
                x = x.cuda()
                
            with torch.no_grad():
                output = model(x)
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                    print(f"  ✅ Features shape: {features.shape}")
                
            print(f"  ✅ Input: {x.shape}")
            print(f"  ✅ Output: {output.shape}")
            
            # 复杂度分析
            complexity = model_complexity_analysis(model, input_size)
            if complexity:
                print(f"  📊 FLOPs: {complexity['flops_str']}")
                print(f"  📊 Params: {complexity['params_str']}")
            
            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  📈 Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    print("🎉 All model tests completed!")
    
    # 模型对比总结
    print("\n📋 Model Comparison Summary:")
    print("├── Standard MainNet: EfficientNet-B4 (pretrained)")
    print("├── Enhanced MainNet: + AMSFE + CLFPF + SALGA")  
    print("└── Super Enhanced: All enhancements enabled")
    print("\n💡 Recommendation: Use Enhanced MainNet with all modules for best performance")
    print("\n📝 Note: Make sure to install efficientnet_pytorch:")
    print("   pip install efficientnet_pytorch")