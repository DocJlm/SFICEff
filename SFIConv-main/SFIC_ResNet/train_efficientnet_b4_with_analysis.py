# train_efficientnet_b4_with_analysis.py - 带模型复杂度分析的训练脚本
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import warnings
import json
from datetime import datetime
import math
warnings.filterwarnings('ignore')

# 导入网络模块
from network.MainNet import create_model
from network.data import SingleInputDataset
from network.utils import setup_seed, cal_metrics
from torchvision import transforms

# 导入模型分析工具
from model_complexity_analyzer import ModelComplexityAnalyzer, quick_model_analysis

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class EfficientNetB4TrainerWithAnalysis:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_metrics = {'acc': 0.0, 'auc': 0.0, 'epoch': 0}
        
        # 设置随机种子
        setup_seed(args.seed)
        
        # 初始化
        self.setup_model()
        self.analyze_model_complexity()  # 新增：模型复杂度分析
        self.setup_data()
        self.setup_optimizer()
        self.setup_logging()
        
    def setup_model(self):
        """设置模型"""
        print("Setting up EfficientNet-B4 Enhanced model...")
        
        # 创建增强版EfficientNet-B4模型
        self.model = create_model(
            model_type='enhanced',
            num_classes=self.args.num_classes,
            drop_rate=self.args.drop_rate,
            enable_amsfe=True,
            enable_clfpf=True,
            enable_salga=True
        )
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            
        self.model = self.model.to(self.device)
        
        print(f"✅ Model setup completed")
        
    def analyze_model_complexity(self):
        """分析模型复杂度"""
        print(f"\n{'='*80}")
        print(f"🔬 PRE-TRAINING MODEL COMPLEXITY ANALYSIS")
        print(f"{'='*80}")
        
        # 创建分析器
        analyzer = ModelComplexityAnalyzer(device=self.device)
        
        # 分析不同配置的模型进行对比
        print(f"\n📊 ANALYZING DIFFERENT MODEL CONFIGURATIONS:")
        print(f"{'-'*80}")
        
        model_configs = [
            {
                'name': 'Baseline EfficientNet-B4',
                'model': create_model(model_type='standard', num_classes=2, drop_rate=0.0)
            },
            {
                'name': 'EfficientNet-B4 + AMSFE',
                'model': create_model(model_type='enhanced', num_classes=2, drop_rate=0.0,
                                    enable_amsfe=True, enable_clfpf=False, enable_salga=False)
            },
            {
                'name': 'EfficientNet-B4 + AMSFE + CLFPF',
                'model': create_model(model_type='enhanced', num_classes=2, drop_rate=0.0,
                                    enable_amsfe=True, enable_clfpf=True, enable_salga=False)
            },
            {
                'name': 'Our Full Model (All Modules)',
                'model': create_model(model_type='enhanced', num_classes=2, drop_rate=0.0,
                                    enable_amsfe=True, enable_clfpf=True, enable_salga=True)
            }
        ]
        
        # 对比分析表格
        print(f"\n{'Model Configuration':<40} {'Params(M)':<12} {'FLOPs(G)':<12} {'Size(MB)':<12} {'Time(ms)':<12}")
        print(f"{'-'*100}")
        
        results = {}
        for config in model_configs:
            try:
                model = config['model'].to(self.device)
                result = quick_model_analysis(model, config['name'])
                results[config['name']] = result
                
                # 格式化输出
                params_m = result['params'] / 1e6
                flops_g = result['flops'] / 1e9 if result['flops'] > 0 else 0
                size_mb = result['size_mb']
                time_ms = result['inference_time_ms']
                
                print(f"{config['name']:<40} {params_m:<12.1f} {flops_g:<12.1f} {size_mb:<12.1f} {time_ms:<12.1f}")
                
            except Exception as e:
                print(f"❌ Error analyzing {config['name']}: {e}")
        
        # 详细分析我们的完整模型
        print(f"\n{'='*80}")
        print(f"🎯 DETAILED ANALYSIS OF OUR ENHANCED MODEL")
        print(f"{'='*80}")
        
        # 获取实际训练模型（可能包含DataParallel）
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 进行详细分析
        detailed_result = analyzer.analyze_model_detailed(
            actual_model, 
            input_size=(1, 3, 224, 224),
            model_name="Enhanced EfficientNet-B4 (Training Model)"
        )
        
        # 与基线模型对比开销分析
        if len(results) >= 2:
            baseline_name = 'Baseline EfficientNet-B4'
            our_name = 'Our Full Model (All Modules)'
            
            if baseline_name in results and our_name in results:
                baseline = results[baseline_name]
                ours = results[our_name]
                
                print(f"\n{'='*80}")
                print(f"📈 ENHANCEMENT MODULES OVERHEAD ANALYSIS")
                print(f"{'='*80}")
                
                param_overhead = (ours['params'] - baseline['params']) / baseline['params'] * 100
                flop_overhead = (ours['flops'] - baseline['flops']) / baseline['flops'] * 100 if baseline['flops'] > 0 else 0
                size_overhead = (ours['size_mb'] - baseline['size_mb']) / baseline['size_mb'] * 100
                time_overhead = (ours['inference_time_ms'] - baseline['inference_time_ms']) / baseline['inference_time_ms'] * 100
                
                print(f"Enhancement Modules Add:")
                print(f"  📊 Parameters:    +{(ours['params'] - baseline['params']):,} (+{param_overhead:.1f}%)")
                print(f"  ⚡ FLOPs:         +{(ours['flops'] - baseline['flops'])/1e9:.1f}G (+{flop_overhead:.1f}%)")
                print(f"  💾 Model Size:    +{(ours['size_mb'] - baseline['size_mb']):.1f}MB (+{size_overhead:.1f}%)")
                print(f"  🚀 Time Overhead: +{(ours['inference_time_ms'] - baseline['inference_time_ms']):.1f}ms (+{time_overhead:.1f}%)")
                
                # 效率评估
                if param_overhead > 0:
                    efficiency_score = 100 / (1 + param_overhead/100 + time_overhead/100)
                    print(f"  💡 Efficiency Score: {efficiency_score:.1f}/100 (higher is better)")
                
                # 性价比分析
                print(f"\n📊 COST-BENEFIT ANALYSIS:")
                print(f"  Each 1% parameter increase costs:")
                if param_overhead > 0:
                    flop_cost_per_param = flop_overhead / param_overhead
                    time_cost_per_param = time_overhead / param_overhead
                    print(f"    - {flop_cost_per_param:.2f}% FLOPs increase")
                    print(f"    - {time_cost_per_param:.2f}% time increase")
        
        # 保存分析结果
        self.save_complexity_analysis(results, detailed_result)
        
        # 内存使用估算
        self.estimate_memory_usage(detailed_result)
        
        print(f"\n{'='*80}")
        print(f"✅ MODEL COMPLEXITY ANALYSIS COMPLETED")
        print(f"{'='*80}")
        
        return detailed_result
    
    def save_complexity_analysis(self, results, detailed_result):
        """保存复杂度分析结果"""
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'model_comparison': results,
            'detailed_analysis': {
                'total_params': detailed_result['parameters']['total_params'],
                'trainable_params': detailed_result['parameters']['trainable_params'],
                'model_size_mb': detailed_result['model_size']['total_size_mb'],
                'flops': detailed_result['thop_analysis']['flops'] if detailed_result['thop_analysis'] else 0,
                'inference_time_ms': detailed_result['timing']['avg_inference_time_ms'],
                'fps': detailed_result['timing']['fps']
            }
        }
        
        # 保存到输出目录
        analysis_file = os.path.join('./output', self.args.name, 'model_complexity_analysis.json')
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"📁 Complexity analysis saved to: {analysis_file}")
    
    def estimate_memory_usage(self, detailed_result):
        """估算训练时内存使用"""
        print(f"\n🧠 MEMORY USAGE ESTIMATION:")
        print(f"{'-'*50}")
        
        # 模型参数内存
        model_memory = detailed_result['model_size']['total_size_mb']
        
        # 梯度内存（与参数相同）
        gradient_memory = model_memory
        
        # 优化器状态内存（AdamW需要2倍参数内存）
        optimizer_memory = model_memory * 2
        
        # 激活内存估算（基于FLOPs和batch size）
        if detailed_result['thop_analysis']:
            # 粗略估算：FLOPs / 1e9 * batch_size * 4 bytes
            activation_memory = detailed_result['thop_analysis']['flops'] / 1e9 * self.args.batch_size * 4 / 1024 / 1024
        else:
            activation_memory = 200  # 默认估算
        
        total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
        
        print(f"Model Parameters:     {model_memory:.1f} MB")
        print(f"Gradients:           {gradient_memory:.1f} MB")
        print(f"Optimizer States:    {optimizer_memory:.1f} MB")
        print(f"Activations (est.):  {activation_memory:.1f} MB")
        print(f"Total Training:      {total_memory:.1f} MB (~{total_memory/1024:.1f} GB)")
        
        # GPU内存检查
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_usage_pct = total_memory / 1024 / gpu_memory * 100
            print(f"GPU Memory:          {gpu_memory:.1f} GB")
            print(f"Estimated Usage:     {memory_usage_pct:.1f}% of GPU memory")
            
            if memory_usage_pct > 90:
                print(f"⚠️  WARNING: High memory usage! Consider reducing batch size.")
            elif memory_usage_pct > 70:
                print(f"⚠️  CAUTION: Moderate memory usage. Monitor during training.")
            else:
                print(f"✅ Good: Memory usage within safe range.")
        
    def setup_data(self):
        """设置数据加载器"""
        print("Setting up data loaders...")
        
        image_size = 224
        
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 训练数据
        train_dataset = SingleInputDataset(
            txt_path=self.args.train_txt_path,
            train_transform=train_transform
        )
        
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # 验证数据
        if os.path.exists(self.args.valid_txt_path):
            val_dataset = SingleInputDataset(
                txt_path=self.args.valid_txt_path,
                valid_transform=val_transform
            )
            self.valid_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
        else:
            print(f"Warning: Validation data not found: {self.args.valid_txt_path}")
            self.valid_loader = None
            
        print(f"Data loaded - Image size: {image_size}x{image_size}")
        print(f"Train samples: {len(train_dataset)}, batches: {len(self.train_loader)}")
        if self.valid_loader:
            print(f"Valid samples: {len(val_dataset)}, batches: {len(self.valid_loader)}")
        
    def setup_optimizer(self):
        """设置优化器"""
        print("Setting up optimizer...")
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 分组参数：预训练骨干 vs 新增模块 vs 分类器
        pretrained_params = []
        enhancement_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            elif any(module_name in name for module_name in ['amsfe', 'clfpf', 'salga']):
                enhancement_params.append(param)
            else:
                pretrained_params.append(param)
        
        print(f"Parameter groups:")
        print(f"  Pretrained backbone: {sum(p.numel() for p in pretrained_params):,} params")
        print(f"  Enhancement modules: {sum(p.numel() for p in enhancement_params):,} params")
        print(f"  Classifier: {sum(p.numel() for p in classifier_params):,} params")
        
        # 使用差分学习率
        param_groups = [
            {'params': pretrained_params, 'lr': self.args.lr * 0.1, 'weight_decay': self.args.weight_decay},
            {'params': enhancement_params, 'lr': self.args.lr * 0.5, 'weight_decay': self.args.weight_decay * 0.5},
            {'params': classifier_params, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay * 0.1},
        ]
        
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False
        )
        
        # 余弦退火调度器 + 预热
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = len(self.train_loader) * 3
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # 使用标签平滑的交叉熵损失
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"Optimizer: AdamW with differential learning rates")
        print(f"Pretrained LR: {self.args.lr * 0.1}")
        print(f"Enhancement LR: {self.args.lr * 0.5}")
        print(f"Classifier LR: {self.args.lr}")
        
    def setup_logging(self):
        """设置日志"""
        self.output_path = os.path.join('./output', self.args.name)
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.output_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        self.log_file = os.path.join(self.output_path, 'training.log')
        print(f"Outputs will be saved to: {self.output_path}")
        
    def log_message(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            try:
                if self.args.use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                # 更新学习率
                self.scheduler.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except RuntimeError as e:
                self.log_message(f"Runtime error in batch {batch_idx}: {e}")
                continue
            
            # 进度输出
            if batch_idx % 50 == 0:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 内存使用监控
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    memory_info = f"GPU: {memory_used:.1f}GB/{memory_cached:.1f}GB"
                else:
                    memory_info = ""
                
                self.log_message(
                    f"Epoch[{epoch+1:2d}/{self.args.epochs}] "
                    f"Batch[{batch_idx:4d}/{len(self.train_loader)}] "
                    f"Loss:{current_loss:.4f} Acc:{current_acc:.2f}% "
                    f"LR:{current_lr:.6f} {memory_info}"
                )
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        if self.valid_loader is None:
            return 0.0, 0.0, 0.0, {}
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.valid_loader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                try:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 收集预测结果用于AUC计算
                    if self.args.num_classes == 2:
                        probs = torch.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        
                except RuntimeError as e:
                    self.log_message(f"Validation error in batch {batch_idx}: {e}")
                    continue
        
        val_loss = running_loss / len(self.valid_loader)
        val_acc = correct / total
        
        # 计算详细指标
        val_auc = 0.0
        detailed_metrics = {}
        
        if self.args.num_classes == 2 and len(set(all_labels)) > 1:
            try:
                val_auc = roc_auc_score(all_labels, all_probs)
                ap_score, auc_score, eer, TPR_2, TPR_3, TPR_4 = cal_metrics(all_labels, all_probs)
                
                detailed_metrics = {
                    'auc': auc_score,
                    'ap': ap_score,
                    'eer': eer,
                    'tpr_2': TPR_2,
                    'tpr_3': TPR_3,
                    'tpr_4': TPR_4
                }
            except Exception as e:
                self.log_message(f"Metrics calculation error: {e}")
        
        return val_loss, val_acc, val_auc, detailed_metrics
    
    def save_checkpoint(self, epoch, is_best=False, extra_info=None):
        """保存检查点"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'args': vars(self.args)
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.output_path, 'latest.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(model_to_save.state_dict(), os.path.join(self.output_path, 'best.pkl'))
            torch.save(checkpoint, os.path.join(self.output_path, 'best.pth'))
    
    def train(self):
        """主训练循环"""
        self.log_message("🚀 Starting EfficientNet-B4 Enhanced training...")
        self.log_message(f"Configuration: {vars(self.args)}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            self.log_message(f"\n{'='*60}")
            self.log_message(f"Epoch {epoch+1}/{self.args.epochs}")
            self.log_message(f"{'='*60}")
            
            try:
                # 训练
                train_loss, train_acc = self.train_epoch(epoch)
                
                # 验证
                val_loss, val_acc, val_auc, detailed_metrics = self.validate_epoch(epoch)
                
                # 记录结果
                self.log_message(f"Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                self.log_message(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
                
                if detailed_metrics:
                    self.log_message(f"Detailed: AP:{detailed_metrics.get('ap', 0):.4f}, EER:{detailed_metrics.get('eer', 0):.4f}")
                
                # 检查是否是最佳模型
                is_best = False
                current_metric = val_auc if self.args.metric == 'auc' else val_acc
                
                if current_metric > self.best_metrics[self.args.metric]:
                    self.best_metrics = {
                        'acc': val_acc,
                        'auc': val_auc,
                        'epoch': epoch,
                        **detailed_metrics
                    }
                    is_best = True
                    patience_counter = 0
                    self.log_message(f"🌟 New best model! {self.args.metric.upper()}: {current_metric:.4f}")
                else:
                    patience_counter += 1
                
                # 保存检查点
                self.save_checkpoint(epoch, is_best=is_best, extra_info={'detailed_metrics': detailed_metrics})
                
                # 早停检查
                if self.args.early_stop > 0 and patience_counter >= self.args.early_stop:
                    self.log_message(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
                    
            except Exception as e:
                self.log_message(f"Error in epoch {epoch+1}: {e}")
                import traceback
                self.log_message(traceback.format_exc())
                continue
        
        # 训练完成
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        self.log_message(f"\n{'='*60}")
        self.log_message(f"🎉 Training completed!")
        self.log_message(f"Best epoch: {self.best_metrics['epoch'] + 1}")
        self.log_message(f"Best accuracy: {self.best_metrics['acc']:.4f}")
        self.log_message(f"Best AUC: {self.best_metrics['auc']:.4f}")
        self.log_message(f"Total time: {hours}h {minutes}m")
        self.log_message(f"{'='*60}")
        
        return self.best_metrics


def main():
    parser = argparse.ArgumentParser(description='EfficientNet-B4 Enhanced Training with Complexity Analysis')
    
    # 基本参数
    parser.add_argument('--name', type=str, default='efficientnet-b4-enhanced-analyzed',
                       help='Experiment name')
    parser.add_argument('--train_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/train.txt',
                       help='Training data txt path')
    parser.add_argument('--valid_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/val.txt',
                       help='Validation data txt path')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 模型参数
    parser.add_argument('--drop_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision training')
    
    # 验证和保存参数
    parser.add_argument('--metric', type=str, default='auc',
                       choices=['acc', 'auc'],
                       help='Metric for model selection')
    parser.add_argument('--early_stop', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    print("🚀 EfficientNet-B4 Enhanced Training with Complexity Analysis")
    print("="*70)
    print(f"Model: EfficientNet-B4 + AMSFE + CLFPF + SALGA")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Mixed precision: {args.use_amp}")
    print("="*70)
    
    try:
        trainer = EfficientNetB4TrainerWithAnalysis(args)
        best_metrics = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"Final best metrics: {best_metrics}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())