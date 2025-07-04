# network/MainNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 只导入EfficientNet模块
try:
    from . import SFIConvEfficientNet_fixed as SFIConvEfficientNet
except ImportError:
    try:
        import SFIConvEfficientNet_fixed as SFIConvEfficientNet
    except ImportError:
        # 如果没有修复版本，使用原版本但可能有问题
        try:
            from . import SFIConvEfficientNet
        except ImportError:
            import SFIConvEfficientNet

# 导入增强模块
try:
    from .enhanced_modules_v2 import AdvancedAMSFEModule, FixedCLFPFModule, SimplifiedSALGAModule
except ImportError:
    try:
        from enhanced_modules_v2 import AdvancedAMSFEModule, FixedCLFPFModule, SimplifiedSALGAModule
    except ImportError:
        print("Warning: Enhanced modules not found, using placeholder modules")
        
        # 占位符模块
        class AdvancedAMSFEModule(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.identity = nn.Identity()
            def forward(self, x):
                return self.identity(x)
        
        class FixedCLFPFModule(nn.Module):
            def __init__(self, in_channels, out_channels=256):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, 1)
            def forward(self, x):
                return self.conv(x)
        
        class SimplifiedSALGAModule(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.identity = nn.Identity()
            def forward(self, x):
                return self.identity(x)


class MainNet(nn.Module):
    """简化的MainNet - 只使用EfficientNet-B4"""
    def __init__(self, num_classes=2, drop_rate=0.2):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        
        # 使用EfficientNet-B4作为唯一骨干
        self.backbone = SFIConvEfficientNet.sfi_efficientnet_b4(
            num_classes=num_classes, 
            drop_rate=drop_rate
        )
        
        print(f"MainNet initialized with SFI-EfficientNet-B4")
        print(f"  - Classes: {num_classes}")
        print(f"  - Drop rate: {drop_rate}")
        
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """提取特征而不进行分类"""
        return self.backbone.extract_features(x)


class EnhancedMainNet(nn.Module):
    """增强版MainNet - 基于EfficientNet-B4"""
    def __init__(self, num_classes=2, drop_rate=0.2,
                 enable_amsfe=True, enable_clfpf=True, enable_salga=True):
        super(EnhancedMainNet, self).__init__()
        self.num_classes = num_classes
        self.enable_amsfe = enable_amsfe
        self.enable_clfpf = enable_clfpf
        self.enable_salga = enable_salga
        
        # EfficientNet-B4骨干网络（不包含分类器）
        backbone_model = SFIConvEfficientNet.sfi_efficientnet_b4(num_classes=num_classes)
        
        # 提取特征部分（去掉分类器）
        self.feature_extractor = nn.Sequential(
            backbone_model.stem,
            *backbone_model.blocks,
            backbone_model.head_conv
        )
        
        # 动态获取特征通道数
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_features = self.feature_extractor(test_input)
            self.feature_channels = test_features.shape[1]
        
        print(f"Detected feature channels: {self.feature_channels}")
        
        current_channels = self.feature_channels
        
        # 增强模块
        if enable_amsfe:
            self.amsfe = AdvancedAMSFEModule(current_channels)
            print(f"AMSFE enabled with {current_channels} channels")
        
        if enable_clfpf:
            self.clfpf = FixedCLFPFModule(current_channels, 512)  # 输出512通道
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
        
        print(f"Enhanced MainNet initialized:")
        print(f"  - Backbone: SFI-EfficientNet-B4")
        print(f"  - Feature flow: {self.feature_channels} -> {current_channels} -> {num_classes}")
        print(f"  - Enhanced modules: AMSFE={enable_amsfe}, CLFPF={enable_clfpf}, SALGA={enable_salga}")
        
    def extract_features(self, x):
        """提取原始骨干特征"""
        features = self.feature_extractor(x)
        # 处理可能的tuple输出
        if isinstance(features, tuple):
            features = features[0]
        return features
        
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
    """超级增强版MainNet - 集成所有先进技术"""
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
        
        # 额外的技术：标签平滑、混合精度等在训练时处理
        print("SuperEnhancedMainNet: All enhancement modules enabled")
        
    def forward(self, x):
        return self.enhanced_net(x)
    
    def extract_features(self, x):
        return self.enhanced_net.extract_features(x)


# 统一的模型创建函数
def create_model(model_type='standard', num_classes=2, drop_rate=0.2, **kwargs):
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


def model_complexity_analysis(model, input_size=(3, 380, 380)):
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
    print("🚀 Testing SFI-EfficientNet-B4 MainNet Models\n")
    
    # 测试不同模型类型
    models_to_test = [
        ('standard', {}),
        ('enhanced', {'enable_amsfe': True, 'enable_clfpf': True, 'enable_salga': False}),
        ('enhanced', {'enable_amsfe': True, 'enable_clfpf': True, 'enable_salga': True}),
        ('super', {})
    ]
    
    input_size = (3, 380, 380)  # EfficientNet-B4推荐尺寸
    batch_size = 2
    
    for model_type, kwargs in models_to_test:
        print(f"Testing {model_type} model...")
        
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
    print("├── Standard MainNet: Pure SFI-EfficientNet-B4")
    print("├── Enhanced MainNet: + AMSFE + CLFPF + SALGA")  
    print("└── Super Enhanced: All enhancements enabled")
    print("\n💡 Recommendation: Use Enhanced MainNet for best performance")