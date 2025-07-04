# train_b4.py - 专门为EfficientNet-B4优化的训练脚本
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
warnings.filterwarnings('ignore')

# 导入网络模块
from network.MainNet import create_model
from network.data import SingleInputDataset, TestDataset
from network.utils import setup_seed, cal_metrics
from torchvision import transforms

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class EfficientNetB4Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_metrics = {'acc': 0.0, 'auc': 0.0, 'epoch': 0}
        
        # 设置随机种子
        setup_seed(args.seed)
        
        # 初始化
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_logging()
        
    def setup_model(self):
        """设置模型"""
        print("Setting up SFI-EfficientNet-B4 model...")
        
        # 创建模型
        self.model = create_model(
            model_type=self.args.model_type,
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
        
        print(f"Model: SFI-EfficientNet-B4 ({self.args.model_type})")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 测试前向传播
        self._test_forward_pass()
        
    def _test_forward_pass(self):
        """测试前向传播"""
        print("Testing forward pass...")
        try:
            # EfficientNet-B4推荐使用380x380
            test_input = torch.randn(2, 3, 380, 380).to(self.device)
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
        
        # EfficientNet-B4推荐使用380x380，但我们从较小尺寸开始训练
        image_size = 320 if self.args.use_large_input else 224
        
        # 创建数据变换
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 训练数据
        if not os.path.exists(self.args.train_txt_path):
            raise FileNotFoundError(f"Training data not found: {self.args.train_txt_path}")
        
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
            drop_last=True
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
                num_workers=8,
                pin_memory=True
            )
        else:
            print(f"Warning: Validation data not found: {self.args.valid_txt_path}")
            self.valid_loader = None
            
        print(f"Data loaded - Image size: {image_size}x{image_size}")
        print(f"Train samples: {len(train_dataset)}, batches: {len(self.train_loader)}")
        if self.valid_loader:
            print(f"Valid samples: {len(val_dataset)}, batches: {len(self.valid_loader)}")
        
    def setup_optimizer(self):
        """设置优化器和调度器"""
        print("Setting up optimizer...")
        
        # 使用不同的学习率for不同组件
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
            
        # 分组参数：骨干网络 vs 分类器
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # 差分学习率
        param_groups = [
            {'params': backbone_params, 'lr': self.args.lr * 0.1},  # 骨干网络用较小学习率
            {'params': classifier_params, 'lr': self.args.lr}        # 分类器用正常学习率
        ]
        
        # 优化器
        if self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
                nesterov=True
            )
        else:
            self.optimizer = optim.Adam(param_groups, weight_decay=self.args.weight_decay)
        
        # 学习率调度器
        if self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.args.epochs,
                eta_min=self.args.lr * 0.01
            )
        elif self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, 
                milestones=[self.args.epochs//3, 2*self.args.epochs//3], 
                gamma=0.1
            )
        else:  # plateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=5, 
                verbose=True,
                min_lr=self.args.lr * 0.001
            )
        
        # 损失函数
        if self.args.use_focal_loss:
            try:
                from network.enhanced_modules_v2 import CombinedLoss
                self.criterion = CombinedLoss()
                print("Using Combined Loss (Focal + CrossEntropy)")
            except ImportError:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                print("Using CrossEntropy Loss with label smoothing")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropy Loss")
        
        print(f"Optimizer: {self.args.optimizer}")
        print(f"Scheduler: {self.args.scheduler}")
        print(f"Backbone LR: {self.args.lr * 0.1}")
        print(f"Classifier LR: {self.args.lr}")
        
    def setup_logging(self):
        """设置日志"""
        self.output_path = os.path.join('./output', self.args.name)
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.output_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        # 日志文件
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
        
        # 进度显示
        total_batches = len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except RuntimeError as e:
                self.log_message(f"Runtime error in batch {batch_idx}: {e}")
                continue
            
            # 进度输出
            if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                progress = 100. * batch_idx / total_batches
                
                self.log_message(
                    f"Epoch[{epoch+1:2d}/{self.args.epochs}] "
                    f"Progress[{progress:5.1f}%] "
                    f"Batch[{batch_idx:4d}/{total_batches}] "
                    f"Loss:{current_loss:.4f} Acc:{current_acc:.2f}%"
                )
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        if self.valid_loader is None:
            return 0.0, 0.0, 0.0
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                try:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 收集预测结果
                    if self.args.num_classes == 2:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                        
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
            
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(self.output_path, f'epoch_{epoch+1}.pth'))
    
    def train(self):
        """主训练循环"""
        self.log_message("🚀 Starting SFI-EfficientNet-B4 training...")
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
                
                # 更新学习率
                if self.args.scheduler == 'plateau':
                    metric_for_scheduler = val_auc if self.args.metric == 'auc' else val_acc
                    self.scheduler.step(metric_for_scheduler)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 记录结果
                self.log_message(f"Training   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                self.log_message(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
                self.log_message(f"Learning Rate: {current_lr:.6f}")
                
                if detailed_metrics:
                    self.log_message(f"Detailed Metrics: {detailed_metrics}")
                
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
                self.save_checkpoint(
                    epoch, 
                    is_best=is_best, 
                    extra_info={'detailed_metrics': detailed_metrics}
                )
                
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
    parser = argparse.ArgumentParser(description='SFI-EfficientNet-B4 Training')
    
    # 基本参数
    parser.add_argument('--name', type=str, default='sfi-efficientnet-b4',
                       help='Experiment name')
    parser.add_argument('--train_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/train.txt',
                       help='Training data txt path')
    parser.add_argument('--valid_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/val.txt',
                       help='Validation data txt path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (reduced for B4)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['standard', 'enhanced', 'super'],
                       help='Model type')
    parser.add_argument('--drop_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--enable_amsfe', action='store_true', default=True,
                       help='Enable AMSFE module')
    parser.add_argument('--enable_clfpf', action='store_true', default=True,
                       help='Enable CLFPF module')
    parser.add_argument('--enable_salga', action='store_true', default=True,
                       help='Enable SALGA module')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='Use focal loss')
    
    # 数据参数
    parser.add_argument('--use_large_input', action='store_true', default=False,
                       help='Use larger input size (320 vs 224)')
    
    # 验证和保存参数
    parser.add_argument('--metric', type=str, default='auc',
                       choices=['acc', 'auc'],
                       help='Metric for model selection')
    parser.add_argument('--early_stop', type=int, default=20,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    print("🚀 SFI-EfficientNet-B4 Training")
    print("="*50)
    print(f"Model: {args.model_type} EfficientNet-B4")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Enhancement modules: AMSFE={args.enable_amsfe}, CLFPF={args.enable_clfpf}, SALGA={args.enable_salga}")
    print("="*50)
    
    try:
        trainer = EfficientNetB4Trainer(args)
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