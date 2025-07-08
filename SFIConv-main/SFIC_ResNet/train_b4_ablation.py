# train_b4_ablation.py - 支持消融实验的EfficientNet-B4训练脚本
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

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class AblationTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_metrics = {'acc': 0.0, 'auc': 0.0, 'epoch': 0}
        
        # 设置随机种子
        setup_seed(args.seed)
        
        # 根据消融实验配置调整实验名称
        self.experiment_name = self._generate_experiment_name()
        
        # 初始化
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_logging()
        
    def _generate_experiment_name(self):
        """根据启用的模块生成实验名称"""
        modules = []
        if self.args.enable_amsfe:
            modules.append("AMSFE")
        if self.args.enable_clfpf:
            modules.append("CLFPF")
        if self.args.enable_salga:
            modules.append("SALGA")
        
        if not modules:
            return f"{self.args.name}_baseline"
        else:
            return f"{self.args.name}_{'_'.join(modules)}"
        
    def setup_model(self):
        """设置模型"""
        print(f"Setting up model for ablation study...")
        print(f"Configuration:")
        print(f"  - AMSFE: {self.args.enable_amsfe}")
        print(f"  - CLFPF: {self.args.enable_clfpf}")
        print(f"  - SALGA: {self.args.enable_salga}")
        
        # 根据消融实验配置创建模型
        if not any([self.args.enable_amsfe, self.args.enable_clfpf, self.args.enable_salga]):
            # 基线模型：使用标准EfficientNet-B4
            print("🔵 Creating BASELINE model (standard EfficientNet-B4)")
            self.model = create_model(
                model_type='standard',
                num_classes=self.args.num_classes,
                drop_rate=self.args.drop_rate
            )
        else:
            # 增强模型：根据参数启用相应模块
            print("🟡 Creating ENHANCED model with selected modules")
            self.model = create_model(
                model_type='enhanced',
                num_classes=self.args.num_classes,
                drop_rate=self.args.drop_rate,
                enable_amsfe=self.args.enable_amsfe,
                enable_clfpf=self.args.enable_clfpf,
                enable_salga=self.args.enable_salga
            )
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            
        self.model = self.model.to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: {self.experiment_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 测试前向传播
        self._test_forward_pass()
        
    def _test_forward_pass(self):
        """测试前向传播"""
        print("Testing forward pass...")
        try:
            test_input = torch.randn(2, 3, 224, 224).to(self.device)
            self.model.eval()
            with torch.no_grad():
                test_output = self.model(test_input)
            self.model.train()
            print(f"✅ Forward pass successful! Output shape: {test_output.shape}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            raise e
        
    def setup_data(self):
        """设置数据加载器"""
        print("Setting up data loaders...")
        
        # 使用224x224作为标准输入尺寸
        image_size = 224
        
        # 优化的数据增强策略
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
        
        # 根据模型类型调整学习率策略
        if not any([self.args.enable_amsfe, self.args.enable_clfpf, self.args.enable_salga]):
            # 基线模型：标准优化策略
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999)
            )
            print("Using standard optimization for baseline model")
        else:
            # 增强模型：差分学习率
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
            
            param_groups = [
                {'params': pretrained_params, 'lr': self.args.lr * 0.1, 'weight_decay': self.args.weight_decay},
                {'params': enhancement_params, 'lr': self.args.lr * 0.5, 'weight_decay': self.args.weight_decay * 0.5},
                {'params': classifier_params, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay * 0.1},
            ]
            
            self.optimizer = optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print("Using differential learning rates for enhanced model")
        
        # 学习率调度器
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = len(self.train_loader) * 3
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"Optimizer: AdamW")
        print(f"Base LR: {self.args.lr}")
        
    def setup_logging(self):
        """设置日志"""
        self.output_path = os.path.join('./output', self.experiment_name)
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存消融实验配置
        ablation_config = {
            'experiment_name': self.experiment_name,
            'enable_amsfe': self.args.enable_amsfe,
            'enable_clfpf': self.args.enable_clfpf,
            'enable_salga': self.args.enable_salga,
            'base_config': vars(self.args)
        }
        
        config_path = os.path.join(self.output_path, 'ablation_config.json')
        with open(config_path, 'w') as f:
            json.dump(ablation_config, f, indent=2)
        
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
                
                self.scheduler.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except RuntimeError as e:
                self.log_message(f"Runtime error in batch {batch_idx}: {e}")
                continue
            
            if batch_idx % 50 == 0:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.log_message(
                    f"Epoch[{epoch+1:2d}/{self.args.epochs}] "
                    f"Batch[{batch_idx:4d}/{len(self.train_loader)}] "
                    f"Loss:{current_loss:.4f} Acc:{current_acc:.2f}% LR:{current_lr:.6f}"
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
                    
                    if self.args.num_classes == 2:
                        probs = torch.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        
                except RuntimeError as e:
                    self.log_message(f"Validation error in batch {batch_idx}: {e}")
                    continue
        
        val_loss = running_loss / len(self.valid_loader)
        val_acc = correct / total
        
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
            'ablation_config': {
                'enable_amsfe': self.args.enable_amsfe,
                'enable_clfpf': self.args.enable_clfpf,
                'enable_salga': self.args.enable_salga
            },
            'args': vars(self.args)
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, os.path.join(self.output_path, 'latest.pth'))
        
        if is_best:
            torch.save(model_to_save.state_dict(), os.path.join(self.output_path, 'best.pkl'))
            torch.save(checkpoint, os.path.join(self.output_path, 'best.pth'))
    
    def train(self):
        """主训练循环"""
        self.log_message(f"🚀 Starting ablation experiment: {self.experiment_name}")
        self.log_message(f"Configuration: AMSFE={self.args.enable_amsfe}, CLFPF={self.args.enable_clfpf}, SALGA={self.args.enable_salga}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            self.log_message(f"\n{'='*60}")
            self.log_message(f"Epoch {epoch+1}/{self.args.epochs}")
            self.log_message(f"{'='*60}")
            
            try:
                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc, val_auc, detailed_metrics = self.validate_epoch(epoch)
                
                self.log_message(f"Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                self.log_message(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
                
                if detailed_metrics:
                    self.log_message(f"Detailed: AP:{detailed_metrics.get('ap', 0):.4f}, EER:{detailed_metrics.get('eer', 0):.4f}")
                
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
                
                self.save_checkpoint(epoch, is_best=is_best, extra_info={'detailed_metrics': detailed_metrics})
                
                if self.args.early_stop > 0 and patience_counter >= self.args.early_stop:
                    self.log_message(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
                    
            except Exception as e:
                self.log_message(f"Error in epoch {epoch+1}: {e}")
                import traceback
                self.log_message(traceback.format_exc())
                continue
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        self.log_message(f"\n{'='*60}")
        self.log_message(f"🎉 Ablation experiment completed: {self.experiment_name}")
        self.log_message(f"Best epoch: {self.best_metrics['epoch'] + 1}")
        self.log_message(f"Best accuracy: {self.best_metrics['acc']:.4f}")
        self.log_message(f"Best AUC: {self.best_metrics['auc']:.4f}")
        self.log_message(f"Total time: {hours}h {minutes}m")
        self.log_message(f"{'='*60}")
        
        return self.best_metrics


def main():
    parser = argparse.ArgumentParser(description='EfficientNet-B4 Ablation Study Training')
    
    # 基本参数
    parser.add_argument('--name', type=str, default='ablation',
                       help='Base experiment name')
    parser.add_argument('--train_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/train.txt',
                       help='Training data txt path')
    parser.add_argument('--valid_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/val.txt',
                       help='Validation data txt path')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 模型参数
    parser.add_argument('--drop_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # 消融实验参数（关键新增）
    parser.add_argument('--enable-amsfe', action='store_true', default=False,
                       help='Enable AMSFE module')
    parser.add_argument('--enable-clfpf', action='store_true', default=False,
                       help='Enable CLFPF module')
    parser.add_argument('--enable-salga', action='store_true', default=False,
                       help='Enable SALGA module')
    
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
    
    print("🚀 EfficientNet-B4 Ablation Study Training")
    print("="*50)
    print(f"Ablation Configuration:")
    print(f"  - AMSFE: {args.enable_amsfe}")
    print(f"  - CLFPF: {args.enable_clfpf}")
    print(f"  - SALGA: {args.enable_salga}")
    print(f"Training Parameters:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print("="*50)
    
    try:
        trainer = AblationTrainer(args)
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