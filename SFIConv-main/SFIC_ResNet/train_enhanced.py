# train_enhanced_fixed.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入网络模块
from network.MainNet import MainNet
from network.enhanced_modules_v2 import SuperEnhancedMainNet, CombinedLoss
from network.data import SingleInputDataset
from network.transform import Data_Transforms
from network.utils import setup_seed, cal_metrics

class FixedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
    def setup_model(self):
        """设置模型"""
        setup_seed(self.args.seed)
        
        if self.args.use_super_enhanced:
            print("Using Super Enhanced MainNet (Fixed Version)")
            self.model = SuperEnhancedMainNet(num_classes=self.args.num_classes)
        else:
            print("Using Original MainNet")
            self.model = MainNet(self.args.num_classes)
            
        self.model = self.model.to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 测试前向传播
        self._test_forward_pass()
        
    def _test_forward_pass(self):
        """测试前向传播"""
        print("Testing forward pass...")
        try:
            test_input = torch.randn(2, 3, 256, 256).to(self.device)
            with torch.no_grad():
                test_output = self.model(test_input)
            print(f"✅ Forward pass successful! Output shape: {test_output.shape}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            raise e
        
    def setup_data(self):
        """设置数据加载器"""
        if not os.path.exists(self.args.train_txt_path):
            print(f"Train data path not found: {self.args.train_txt_path}")
            return
            
        self.train_data = SingleInputDataset(
            txt_path=self.args.train_txt_path, 
            train_transform=Data_Transforms['train']
        )
        self.train_loader = DataLoader(
            dataset=self.train_data, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        if os.path.exists(self.args.valid_txt_path):
            self.valid_data = SingleInputDataset(
                txt_path=self.args.valid_txt_path, 
                valid_transform=Data_Transforms['val']
            )
            self.valid_loader = DataLoader(
                dataset=self.valid_data, 
                batch_size=self.args.batch_size, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True
            )
        else:
            print(f"Valid data path not found: {self.args.valid_txt_path}")
            
        print(f"Train data: {len(self.train_data)} samples")
        print(f"Valid data: {len(self.valid_data)} samples")
        
    def setup_training(self):
        """设置训练组件"""
        # 损失函数
        if self.args.use_focal_loss:
            self.criterion = CombinedLoss()
            print("Using Combined Loss (Focal + CrossEntropy)")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropy Loss")
            
        # 优化器
        if self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
            
        # 学习率调度器
        if self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
            
        # 创建输出目录
        self.output_path = os.path.join('./output', self.args.name)
        os.makedirs(self.output_path, exist_ok=True)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            except RuntimeError as e:
                print(f"Runtime error in batch {batch_idx}: {e}")
                print(f"Input shape: {images.shape}, Labels shape: {labels.shape}")
                raise e
            
            # 输出训练进度
            if batch_idx % 200 == 0:  # 减少输出频率
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                print(f"Training: Epoch[{epoch+1:2d}/{self.args.epochs}] "
                      f"Batch[{batch_idx:4d}/{len(self.train_loader)}] "
                      f"Loss:{current_loss:.4f} Acc:{current_acc:.2f}%")
                      
        return running_loss / len(self.train_loader), correct / total
        
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                try:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 收集用于AUC计算的数据
                    if self.args.num_classes == 2:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        
                except RuntimeError as e:
                    print(f"Validation error: {e}")
                    continue
                    
        val_acc = correct / total
        val_loss = running_loss / len(self.valid_loader)
        
        # 计算AUC
        val_auc = 0.0
        if self.args.num_classes == 2 and len(set(all_labels)) > 1:
            try:
                val_auc = roc_auc_score(all_labels, all_probs)
            except:
                val_auc = 0.0
                
        return val_loss, val_acc, val_auc
        
    def train(self):
        """主训练循环"""
        print("Starting training...")
        start_time = time.time()
        
        best_acc = 0.0
        best_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print('-' * 60)
            
            try:
                # 训练
                train_loss, train_acc = self.train_epoch(epoch)
                
                # 验证
                val_loss, val_acc, val_auc = self.validate_epoch(epoch)
                
                # 更新学习率
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 输出结果
                print(f"Validating: Epoch[{epoch+1:2d}/{self.args.epochs}] "
                      f"Loss:{val_loss:.4f} Acc:{val_acc:.2%} AUC:{val_auc:.4f} "
                      f"LR:{current_lr:.6f}")
                
                # 保存最佳模型
                is_best = False
                if self.args.metric == 'auc':
                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_acc = val_acc
                        best_epoch = epoch
                        is_best = True
                        patience_counter = 0
                    else:
                        patience_counter += 1
                else:
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_auc = val_auc
                        best_epoch = epoch
                        is_best = True
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                if is_best:
                    torch.save(self.model.state_dict(), 
                              os.path.join(self.output_path, "best.pkl"))
                    print(f"★ New best model saved! Best {self.args.metric}: {best_auc if self.args.metric == 'auc' else best_acc:.4f}")
                    
                # 早停检查
                if self.args.early_stop > 0 and patience_counter >= self.args.early_stop:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
                    
                # 定期保存
                if (epoch + 1) % 10 == 0:
                    torch.save(self.model.state_dict(), 
                              os.path.join(self.output_path, f"epoch_{epoch+1}.pth"))
                              
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                continue
                
        # 训练完成
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best accuracy: {best_acc:.4f}")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Total time: {total_time:.2f}s")
        print("="*60)
        
        return best_acc, best_auc

def main():
    parser = argparse.ArgumentParser()
    
    # 基本参数
    parser.add_argument('--name', type=str, default='super-enhanced-fixed')
    parser.add_argument('--train_txt_path', type=str, default='/home/zqc/FaceForensics++/c23/train.txt')
    parser.add_argument('--valid_txt_path', type=str, default='/home/zqc/FaceForensics++/c23/val.txt')
    parser.add_argument('--batch_size', type=int, default=16)  # 更保守的batch size
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    
    # 模型参数
    parser.add_argument('--use_super_enhanced', action='store_true', default=True,
                       help='Use super enhanced model')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=5e-5)  # 更小的学习率
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_focal_loss', action='store_true', default=True)
    
    # 验证和保存参数
    parser.add_argument('--metric', type=str, default='auc', choices=['acc', 'auc'])
    parser.add_argument('--early_stop', type=int, default=10)
    
    args = parser.parse_args()
    
    print("🚀 Starting Fixed Super Enhanced Training")
    print("="*50)
    print(f"Configuration:")
    print(f"  - Model: Super Enhanced MainNet (Fixed)")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Optimizer: {args.optimizer}")
    print(f"  - Use focal loss: {args.use_focal_loss}")
    print("="*50)
    
    try:
        # 创建并运行训练器
        trainer = FixedTrainer(args)
        best_acc, best_auc = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"Final results: Acc={best_acc:.4f}, AUC={best_auc:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == '__main__':
    main()