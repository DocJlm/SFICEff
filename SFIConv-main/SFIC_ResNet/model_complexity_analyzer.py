# model_complexity_analyzer.py - 模型复杂度分析工具
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import time
import os

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("⚠️  thop not installed. Install with: pip install thop")
    THOP_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    print("⚠️  ptflops not installed. Install with: pip install ptflops")
    PTFLOPS_AVAILABLE = False

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    print("⚠️  fvcore not installed. Install with: pip install fvcore")
    FVCORE_AVAILABLE = False

class ModelComplexityAnalyzer:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def count_parameters(self, model):
        """统计模型参数"""
        total_params = 0
        trainable_params = 0
        
        param_details = {}
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            # 按模块分组统计
            module_name = name.split('.')[0] if '.' in name else 'other'
            if module_name not in param_details:
                param_details[module_name] = {'total': 0, 'trainable': 0}
            
            param_details[module_name]['total'] += param_count
            if param.requires_grad:
                param_details[module_name]['trainable'] += param_count
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'module_details': param_details
        }
    
    def calculate_model_size(self, model):
        """计算模型大小"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = param_size + buffer_size
        model_size_mb = model_size / 1024 / 1024
        
        return {
            'param_size_bytes': param_size,
            'buffer_size_bytes': buffer_size,
            'total_size_bytes': model_size,
            'total_size_mb': model_size_mb
        }
    
    def profile_with_thop(self, model, input_size=(1, 3, 224, 224)):
        """使用thop进行性能分析"""
        if not THOP_AVAILABLE:
            return None
        
        model_copy = model.eval()
        dummy_input = torch.randn(input_size).to(self.device)
        
        try:
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
            flops_str, params_str = clever_format([flops, params], "%.3f")
            
            return {
                'flops': flops,
                'params': params,
                'flops_str': flops_str,
                'params_str': params_str
            }
        except Exception as e:
            print(f"THOP analysis failed: {e}")
            return None
    
    def profile_with_ptflops(self, model, input_size=(3, 224, 224)):
        """使用ptflops进行性能分析"""
        if not PTFLOPS_AVAILABLE:
            return None
        
        try:
            model_copy = model.eval()
            flops, params = get_model_complexity_info(
                model_copy, 
                input_size, 
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False
            )
            
            return {
                'flops_str': flops,
                'params_str': params
            }
        except Exception as e:
            print(f"ptflops analysis failed: {e}")
            return None
    
    def profile_with_fvcore(self, model, input_size=(1, 3, 224, 224)):
        """使用fvcore进行性能分析"""
        if not FVCORE_AVAILABLE:
            return None
        
        try:
            model_copy = model.eval()
            dummy_input = torch.randn(input_size).to(self.device)
            
            flop_dict, _ = flop_count(model_copy, (dummy_input,), mode=FlopCountMode.TABLE)
            total_flops = sum(flop_dict.values())
            
            return {
                'flops': total_flops,
                'flop_details': flop_dict
            }
        except Exception as e:
            print(f"fvcore analysis failed: {e}")
            return None
    
    def measure_inference_time(self, model, input_size=(1, 3, 224, 224), num_runs=100):
        """测量推理时间"""
        model = model.eval().to(self.device)
        dummy_input = torch.randn(input_size).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测量时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        fps = 1000 / avg_time
        
        return {
            'avg_inference_time_ms': avg_time,
            'fps': fps,
            'total_time_ms': (end_time - start_time) * 1000,
            'num_runs': num_runs
        }
    
    def analyze_model_detailed(self, model, input_size=(1, 3, 224, 224), model_name="Model"):
        """详细分析模型"""
        print(f"\n{'='*80}")
        print(f"🔬 DETAILED MODEL ANALYSIS: {model_name}")
        print(f"{'='*80}")
        
        model = model.to(self.device)
        
        # 1. 参数统计
        print(f"\n📊 PARAMETER ANALYSIS:")
        print(f"{'-'*50}")
        param_info = self.count_parameters(model)
        
        print(f"Total Parameters:      {param_info['total_params']:,}")
        print(f"Trainable Parameters:  {param_info['trainable_params']:,}")
        print(f"Non-trainable Params:  {param_info['non_trainable_params']:,}")
        print(f"Trainable Ratio:       {param_info['trainable_params']/param_info['total_params']*100:.2f}%")
        
        # 按模块详细统计
        print(f"\n📋 Parameters by Module:")
        for module_name, details in param_info['module_details'].items():
            percentage = details['total'] / param_info['total_params'] * 100
            print(f"  {module_name:20s}: {details['total']:>10,} ({percentage:5.1f}%) "
                  f"[Trainable: {details['trainable']:,}]")
        
        # 2. 模型大小
        print(f"\n💾 MODEL SIZE ANALYSIS:")
        print(f"{'-'*50}")
        size_info = self.calculate_model_size(model)
        print(f"Model Size:           {size_info['total_size_mb']:.2f} MB")
        print(f"Parameter Size:       {size_info['param_size_bytes']/1024/1024:.2f} MB")
        print(f"Buffer Size:          {size_info['buffer_size_bytes']/1024/1024:.2f} MB")
        
        # 3. FLOPs分析
        print(f"\n⚡ COMPUTATIONAL COMPLEXITY:")
        print(f"{'-'*50}")
        
        # THOP分析
        thop_result = self.profile_with_thop(model, input_size)
        if thop_result:
            print(f"THOP Analysis:")
            print(f"  FLOPs:              {thop_result['flops_str']}")
            print(f"  Parameters:         {thop_result['params_str']}")
            print(f"  Raw FLOPs:          {thop_result['flops']:,}")
        
        # ptflops分析
        ptflops_result = self.profile_with_ptflops(model, input_size[1:])
        if ptflops_result:
            print(f"ptflops Analysis:")
            print(f"  FLOPs:              {ptflops_result['flops_str']}")
            print(f"  Parameters:         {ptflops_result['params_str']}")
        
        # fvcore分析
        fvcore_result = self.profile_with_fvcore(model, input_size)
        if fvcore_result:
            print(f"fvcore Analysis:")
            print(f"  FLOPs:              {fvcore_result['flops']:,}")
        
        # 4. 推理性能
        print(f"\n🚀 INFERENCE PERFORMANCE:")
        print(f"{'-'*50}")
        timing_info = self.measure_inference_time(model, input_size)
        print(f"Average Inference Time: {timing_info['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput (FPS):       {timing_info['fps']:.1f}")
        print(f"Device:                 {self.device}")
        print(f"Input Size:             {input_size}")
        
        # 5. 效率分析
        print(f"\n📈 EFFICIENCY METRICS:")
        print(f"{'-'*50}")
        
        if thop_result:
            flops_per_param = thop_result['flops'] / param_info['total_params']
            print(f"FLOPs per Parameter:    {flops_per_param:.2f}")
            
            # 内存效率（FLOPs per MB）
            flops_per_mb = thop_result['flops'] / size_info['total_size_mb']
            print(f"FLOPs per MB:           {flops_per_mb/1e6:.2f} M")
            
            # 速度效率（FLOPs per ms）
            flops_per_ms = thop_result['flops'] / timing_info['avg_inference_time_ms']
            print(f"FLOPs per ms:           {flops_per_ms/1e6:.2f} M")
        
        # 6. 与EfficientNet基线对比
        self.compare_with_baselines(param_info, thop_result, size_info, timing_info)
        
        return {
            'parameters': param_info,
            'model_size': size_info,
            'thop_analysis': thop_result,
            'ptflops_analysis': ptflops_result,
            'fvcore_analysis': fvcore_result,
            'timing': timing_info
        }
    
    def compare_with_baselines(self, param_info, thop_result, size_info, timing_info):
        """与基线模型对比"""
        print(f"\n📊 COMPARISON WITH BASELINES:")
        print(f"{'-'*50}")
        
        # 基线模型数据（近似值）
        baselines = {
            'EfficientNet-B4': {'params': 19.3e6, 'flops': 4.2e9, 'size_mb': 77.1},
            'ResNet-50': {'params': 25.6e6, 'flops': 4.1e9, 'size_mb': 102.4},
            'ResNet-101': {'params': 44.5e6, 'flops': 7.8e9, 'size_mb': 178.0},
        }
        
        print(f"{'Model':<20} {'Params (M)':<12} {'FLOPs (G)':<12} {'Size (MB)':<12} {'Efficiency':<12}")
        print(f"{'-'*70}")
        
        # 我们的模型
        our_params_m = param_info['total_params'] / 1e6
        our_flops_g = thop_result['flops'] / 1e9 if thop_result else 0
        our_size_mb = size_info['total_size_mb']
        our_efficiency = our_flops_g / our_params_m if our_params_m > 0 else 0
        
        print(f"{'Our Model':<20} {our_params_m:<12.1f} {our_flops_g:<12.1f} {our_size_mb:<12.1f} {our_efficiency:<12.1f}")
        
        # 基线模型
        for name, specs in baselines.items():
            baseline_params_m = specs['params'] / 1e6
            baseline_flops_g = specs['flops'] / 1e9
            baseline_size_mb = specs['size_mb']
            baseline_efficiency = baseline_flops_g / baseline_params_m
            
            print(f"{name:<20} {baseline_params_m:<12.1f} {baseline_flops_g:<12.1f} {baseline_size_mb:<12.1f} {baseline_efficiency:<12.1f}")
        
        # 相对比较
        if thop_result:
            print(f"\n📈 Relative to EfficientNet-B4:")
            eff_b4 = baselines['EfficientNet-B4']
            param_ratio = param_info['total_params'] / eff_b4['params']
            flop_ratio = thop_result['flops'] / eff_b4['flops']
            size_ratio = size_info['total_size_mb'] / eff_b4['size_mb']
            
            print(f"  Parameters: {param_ratio:.2f}x ({'↑' if param_ratio > 1 else '↓'} {abs(param_ratio-1)*100:.1f}%)")
            print(f"  FLOPs:      {flop_ratio:.2f}x ({'↑' if flop_ratio > 1 else '↓'} {abs(flop_ratio-1)*100:.1f}%)")
            print(f"  Size:       {size_ratio:.2f}x ({'↑' if size_ratio > 1 else '↓'} {abs(size_ratio-1)*100:.1f}%)")
    
    def analyze_enhancement_modules(self, model_standard, model_enhanced):
        """分析增强模块的开销"""
        print(f"\n{'='*80}")
        print(f"🔍 ENHANCEMENT MODULES IMPACT ANALYSIS")
        print(f"{'='*80}")
        
        # 分析标准模型
        standard_info = self.count_parameters(model_standard)
        standard_size = self.calculate_model_size(model_standard)
        standard_thop = self.profile_with_thop(model_standard)
        standard_timing = self.measure_inference_time(model_standard)
        
        # 分析增强模型
        enhanced_info = self.count_parameters(model_enhanced)
        enhanced_size = self.calculate_model_size(model_enhanced)
        enhanced_thop = self.profile_with_thop(model_enhanced)
        enhanced_timing = self.measure_inference_time(model_enhanced)
        
        # 计算增强模块的开销
        added_params = enhanced_info['total_params'] - standard_info['total_params']
        added_size = enhanced_size['total_size_mb'] - standard_size['total_size_mb']
        
        if standard_thop and enhanced_thop:
            added_flops = enhanced_thop['flops'] - standard_thop['flops']
            flop_increase = (enhanced_thop['flops'] / standard_thop['flops'] - 1) * 100
        else:
            added_flops = 0
            flop_increase = 0
        
        param_increase = (enhanced_info['total_params'] / standard_info['total_params'] - 1) * 100
        size_increase = (enhanced_size['total_size_mb'] / standard_size['total_size_mb'] - 1) * 100
        time_increase = (enhanced_timing['avg_inference_time_ms'] / standard_timing['avg_inference_time_ms'] - 1) * 100
        
        print(f"\n📊 ENHANCEMENT OVERHEAD:")
        print(f"{'-'*50}")
        print(f"Added Parameters:       {added_params:,} (+{param_increase:.1f}%)")
        print(f"Added Model Size:       {added_size:.2f} MB (+{size_increase:.1f}%)")
        if added_flops > 0:
            print(f"Added FLOPs:            {added_flops:,} (+{flop_increase:.1f}%)")
        print(f"Time Overhead:          {enhanced_timing['avg_inference_time_ms'] - standard_timing['avg_inference_time_ms']:.2f} ms (+{time_increase:.1f}%)")
        
        # 模块效率分析
        print(f"\n💡 ENHANCEMENT EFFICIENCY:")
        print(f"{'-'*50}")
        if added_params > 0:
            efficiency_score = param_increase / (param_increase + flop_increase + time_increase) * 100
            print(f"Parameter Efficiency:   {standard_info['total_params']/added_params:.1f}x baseline per added param")
            print(f"Overall Efficiency:     {efficiency_score:.1f}% (lower overhead is better)")
        
        return {
            'standard': {'params': standard_info, 'size': standard_size, 'flops': standard_thop, 'timing': standard_timing},
            'enhanced': {'params': enhanced_info, 'size': enhanced_size, 'flops': enhanced_thop, 'timing': enhanced_timing},
            'overhead': {
                'added_params': added_params,
                'param_increase_pct': param_increase,
                'added_size_mb': added_size,
                'size_increase_pct': size_increase,
                'added_flops': added_flops,
                'flop_increase_pct': flop_increase,
                'time_increase_pct': time_increase
            }
        }

def analyze_all_model_variants():
    """分析所有模型变体"""
    from network.MainNet import create_model
    
    analyzer = ModelComplexityAnalyzer()
    
    print(f"🔬 COMPREHENSIVE MODEL ANALYSIS")
    print(f"{'='*80}")
    
    # 分析不同配置的模型
    model_configs = [
        {'name': 'Baseline EfficientNet-B4', 'type': 'standard', 'kwargs': {}},
        {'name': 'EfficientNet-B4 + AMSFE', 'type': 'enhanced', 'kwargs': {'enable_amsfe': True, 'enable_clfpf': False, 'enable_salga': False}},
        {'name': 'EfficientNet-B4 + AMSFE + CLFPF', 'type': 'enhanced', 'kwargs': {'enable_amsfe': True, 'enable_clfpf': True, 'enable_salga': False}},
        {'name': 'EfficientNet-B4 + All Modules', 'type': 'enhanced', 'kwargs': {'enable_amsfe': True, 'enable_clfpf': True, 'enable_salga': True}},
    ]
    
    all_results = {}
    
    for config in model_configs:
        try:
            print(f"\n{'='*80}")
            print(f"Analyzing: {config['name']}")
            
            model = create_model(
                model_type=config['type'],
                num_classes=2,
                drop_rate=0.0,  # 用于分析的模型不使用dropout
                **config['kwargs']
            )
            
            result = analyzer.analyze_model_detailed(model, model_name=config['name'])
            all_results[config['name']] = result
            
        except Exception as e:
            print(f"❌ Error analyzing {config['name']}: {e}")
    
    # 生成对比表格
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE COMPARISON TABLE")
    print(f"{'='*80}")
    
    print(f"{'Model':<35} {'Params(M)':<12} {'FLOPs(G)':<12} {'Size(MB)':<12} {'Time(ms)':<12} {'FPS':<8}")
    print(f"{'-'*95}")
    
    for name, result in all_results.items():
        params_m = result['parameters']['total_params'] / 1e6
        flops_g = result['thop_analysis']['flops'] / 1e9 if result['thop_analysis'] else 0
        size_mb = result['model_size']['total_size_mb']
        time_ms = result['timing']['avg_inference_time_ms']
        fps = result['timing']['fps']
        
        print(f"{name:<35} {params_m:<12.1f} {flops_g:<12.1f} {size_mb:<12.1f} {time_ms:<12.1f} {fps:<8.1f}")
    
    return all_results

# 在训练前调用的简化分析函数
def quick_model_analysis(model, model_name="Model"):
    """训练前快速模型分析"""
    analyzer = ModelComplexityAnalyzer()
    
    print(f"\n🔬 Quick Model Analysis: {model_name}")
    print(f"{'='*60}")
    
    # 参数统计
    param_info = analyzer.count_parameters(model)
    size_info = analyzer.calculate_model_size(model)
    
    print(f"📊 Parameters:      {param_info['total_params']:,} ({param_info['total_params']/1e6:.1f}M)")
    print(f"💾 Model Size:      {size_info['total_size_mb']:.1f} MB")
    
    # FLOPs分析
    thop_result = analyzer.profile_with_thop(model)
    if thop_result:
        print(f"⚡ FLOPs:           {thop_result['flops_str']} ({thop_result['flops']/1e9:.1f}G)")
        print(f"📈 Efficiency:      {thop_result['flops']/param_info['total_params']:.1f} FLOPs/param")
    
    # 推理时间
    timing_info = analyzer.measure_inference_time(model, num_runs=10)
    print(f"🚀 Inference Time:  {timing_info['avg_inference_time_ms']:.1f} ms")
    print(f"📺 Throughput:      {timing_info['fps']:.1f} FPS")
    
    print(f"{'='*60}")
    
    return {
        'params': param_info['total_params'],
        'size_mb': size_info['total_size_mb'],
        'flops': thop_result['flops'] if thop_result else 0,
        'inference_time_ms': timing_info['avg_inference_time_ms'],
        'fps': timing_info['fps']
    }

if __name__ == "__main__":
    # 运行完整分析
    analyze_all_model_variants()