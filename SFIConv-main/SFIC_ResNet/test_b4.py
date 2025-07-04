# test_b4.py - 专门为EfficientNet-B4优化的测试脚本
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# 导入模块
from network.MainNet import create_model
from network.data import SingleInputDataset, TestDataset, TTADataset
from network.utils import cal_metrics
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class EfficientNetB4Tester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        """设置模型"""
        print(f"Setting up {self.args.model_type} SFI-EfficientNet-B4...")
        
        # 创建模型
        self.model = create_model(
            model_type=self.args.model_type,
            num_classes=self.args.num_classes,
            drop_rate=0.0,  # 测试时不使用dropout
            enable_amsfe=self.args.enable_amsfe,
            enable_clfpf=self.args.enable_clfpf,
            enable_salga=self.args.enable_salga
        )
        
        # 加载模型权重
        if os.path.exists(self.args.model_path):
            print(f"Loading model from: {self.args.model_path}")
            self.load_model()
        else:
            raise FileNotFoundError(f"Model not found: {self.args.model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded successfully! Parameters: {total_params:,}")
        
    def load_model(self):
        """加载模型权重"""
        try:
            # 尝试加载完整检查点
            checkpoint = torch.load(self.args.model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 完整检查点
                state_dict = checkpoint['model_state_dict']
                print("Loaded from checkpoint")
            else:
                # 只有state_dict
                state_dict = checkpoint
                print("Loaded state dict only")
            
            # 处理DataParallel保存的模型
            if any(key.startswith('module.') for key in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            # 加载权重
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_data(self):
        """设置数据加载器"""
        print("Setting up test data...")
        
        if not os.path.exists(self.args.test_txt_path):
            raise FileNotFoundError(f"Test data not found: {self.args.test_txt_path}")
        
        # 创建测试变换 (EfficientNet-B4推荐380x380)
        image_size = 380
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 标准测试数据加载器
        test_dataset = TestDataset(
            txt_path=self.args.test_txt_path,
            test_transform=test_transform
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        # TTA数据加载器（如果启用）
        if self.args.use_tta:
            self.tta_dataset = TTADataset(
                txt_path=self.args.test_txt_path,
                backbone='efficientnet',
                backbone_config='b4'
            )
            self.tta_loader = DataLoader(
                self.tta_dataset,
                batch_size=8,  # TTA使用较小的batch size
                shuffle=False,
                num_workers=4
            )
            print(f"TTA enabled with {len(self.tta_dataset)} samples")
        
        print(f"Test data loaded: {len(test_dataset)} samples")
    
    def test_standard(self):
        """标准测试"""
        print("Running standard test...")
        
        all_labels = []
        all_probs = []
        all_preds = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                try:
                    # 前向传播
                    outputs = self.model(images)
                    
                    # 预测
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 收集结果
                    if self.args.num_classes == 2:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
                
                # 进度显示
                if batch_idx % 100 == 0:
                    current_acc = correct / total if total > 0 else 0
                    print(f"Batch [{batch_idx:4d}/{len(self.test_loader)}] Acc: {current_acc:.2%}")
        
        return self.calculate_metrics(all_labels, all_probs, all_preds, correct, total)
    
    def test_with_tta(self):
        """使用测试时数据增强"""
        print("Running test with TTA...")
        
        all_labels = []
        all_probs = []
        all_preds = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (augmented_images, labels) in enumerate(self.tta_loader):
                labels = labels.to(self.device)
                
                # 对每个增强版本进行预测
                tta_outputs = []
                for aug_imgs in augmented_images:
                    aug_imgs = aug_imgs.to(self.device)
                    outputs = self.model(aug_imgs)
                    tta_outputs.append(outputs)
                
                # 平均所有增强版本的输出
                avg_outputs = torch.mean(torch.stack(tta_outputs), dim=0)
                
                # 预测
                _, predicted = torch.max(avg_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 收集结果
                if self.args.num_classes == 2:
                    probs = torch.nn.functional.softmax(avg_outputs, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    current_acc = correct / total if total > 0 else 0
                    print(f"TTA Batch [{batch_idx:4d}/{len(self.tta_loader)}] Acc: {current_acc:.2%}")
        
        return self.calculate_metrics(all_labels, all_probs, all_preds, correct, total)
    
    def calculate_metrics(self, all_labels, all_probs, all_preds, correct, total):
        """计算详细指标"""
        results = {
            'accuracy': correct / total if total > 0 else 0,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        if self.args.num_classes == 2 and len(all_labels) > 0:
            try:
                # 基本指标
                ap_score, auc_score, eer, TPR_2, TPR_3, TPR_4 = cal_metrics(all_labels, all_probs)
                
                results.update({
                    'auc': auc_score,
                    'ap': ap_score,
                    'eer': eer,
                    'tpr_at_fpr_1e-2': TPR_2,
                    'tpr_at_fpr_1e-3': TPR_3,
                    'tpr_at_fpr_1e-4': TPR_4
                })
                
                # 混淆矩阵
                cm = confusion_matrix(all_labels, all_preds)
                results['confusion_matrix'] = cm.tolist()
                
                # 分类报告的各项指标
                tn, fp, fn, tp = cm.ravel()
                results.update({
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp),
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
                })
                
                # F1 score
                precision = results['precision']
                recall = results['recall']
                results['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # 为可视化保存数据
                results['labels'] = all_labels
                results['probs'] = all_probs
                results['predictions'] = all_preds
                
            except Exception as e:
                print(f"Error calculating detailed metrics: {e}")
        
        return results
    
    def save_results(self, results, suffix=""):
        """保存测试结果"""
        output_dir = os.path.dirname(self.args.model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存文本结果
        result_file = os.path.join(output_dir, f'test_results{suffix}.txt')
        with open(result_file, 'w') as f:
            f.write(f"SFI-EfficientNet-B4 Test Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.args.model_type}\n")
            f.write(f"Model path: {self.args.model_path}\n")
            f.write(f"Test data: {self.args.test_txt_path}\n")
            f.write(f"Use TTA: {self.args.use_tta}\n")
            f.write(f"\nResults:\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            
            if 'auc' in results:
                f.write(f"AUC: {results['auc']:.4f}\n")
                f.write(f"AP: {results['ap']:.4f}\n")
                f.write(f"EER: {results['eer']:.4f}\n")
                f.write(f"F1 Score: {results['f1_score']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"Specificity: {results['specificity']:.4f}\n")
                f.write(f"TPR@FPR=1e-2: {results['tpr_at_fpr_1e-2']:.4f}\n")
                f.write(f"TPR@FPR=1e-3: {results['tpr_at_fpr_1e-3']:.4f}\n")
                f.write(f"TPR@FPR=1e-4: {results['tpr_at_fpr_1e-4']:.4f}\n")
        
        # 保存JSON结果（去掉原始数据）
        json_results = {k: v for k, v in results.items() 
                       if k not in ['labels', 'probs', 'predictions']}
        json_file = os.path.join(output_dir, f'test_results{suffix}.json')
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        return result_file
    
    def plot_results(self, results, suffix=""):
        """绘制结果图表"""
        if 'labels' not in results or 'probs' not in results:
            return
        
        output_dir = os.path.dirname(self.args.model_path)
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(results['labels'], results['probs'])
        auc_score = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, 'b-', label=f'AUC = {auc_score:.4f}')
        ax1.plot([0, 1], [0, 1], 'r--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # 混淆矩阵
        cm = confusion_matrix(results['labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 概率分布
        real_probs = [results['probs'][i] for i, label in enumerate(results['labels']) if label == 0]
        fake_probs = [results['probs'][i] for i, label in enumerate(results['labels']) if label == 1]
        
        ax3.hist(real_probs, bins=50, alpha=0.7, label='Real', color='green')
        ax3.hist(fake_probs, bins=50, alpha=0.7, label='Fake', color='red')
        ax3.set_xlabel('Prediction Probability')
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 性能指标条形图
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score']
        values = [results['accuracy'], results['auc'], results['precision'], 
                 results['recall'], results['f1_score']]
        
        bars = ax4.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
        ax4.set_ylim([0, 1])
        ax4.set_title('Performance Metrics')
        ax4.set_ylabel('Score')
        
        # 在条形图上添加数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(output_dir, f'test_results{suffix}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {plot_file}")
    
    def run_test(self):
        """运行测试"""
        print(f"\n🚀 Testing SFI-EfficientNet-B4 ({self.args.model_type})")
        print("="*60)
        
        # 标准测试
        print("Running standard inference...")
        standard_results = self.test_standard()
        
        # 保存标准测试结果
        self.save_results(standard_results, "_standard")
        self.plot_results(standard_results, "_standard")
        
        # TTA测试（如果启用）
        if self.args.use_tta:
            print("\nRunning test with TTA...")
            tta_results = self.test_with_tta()
            self.save_results(tta_results, "_tta")
            self.plot_results(tta_results, "_tta")
            
            # 比较结果
            print("\n📊 Results Comparison:")
            print(f"Standard - Acc: {standard_results['accuracy']:.4f}, AUC: {standard_results.get('auc', 0):.4f}")
            print(f"TTA      - Acc: {tta_results['accuracy']:.4f}, AUC: {tta_results.get('auc', 0):.4f}")
            
            improvement = tta_results.get('auc', 0) - standard_results.get('auc', 0)
            print(f"TTA Improvement: {improvement:+.4f}")
            
            return tta_results
        
        return standard_results


def main():
    parser = argparse.ArgumentParser(description='SFI-EfficientNet-B4 Testing')
    
    # 基本参数
    parser.add_argument('--test_txt_path', type=str, 
                       default='/home/zqc/FaceForensics++/c23/test.txt',
                       help='Test data txt path')
    parser.add_argument('--model_path', type=str, 
                       default='./output/sfi-efficientnet-b4-enhanced/best.pkl',
                       help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['standard', 'enhanced', 'super'],
                       help='Model type')
    parser.add_argument('--enable_amsfe', action='store_true', default=True,
                       help='Enable AMSFE module')
    parser.add_argument('--enable_clfpf', action='store_true', default=True,
                       help='Enable CLFPF module')
    parser.add_argument('--enable_salga', action='store_true', default=True,
                       help='Enable SALGA module')
    
    # 测试选项
    parser.add_argument('--use_tta', action='store_true', default=False,
                       help='Use test time augmentation')
    
    args = parser.parse_args()
    
    print("🧪 SFI-EfficientNet-B4 Testing")
    print("="*40)
    print(f"Model: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_txt_path}")
    print(f"Use TTA: {args.use_tta}")
    print("="*40)
    
    try:
        tester = EfficientNetB4Tester(args)
        results = tester.run_test()
        
        print("\n🎉 Testing completed successfully!")
        print(f"Final Accuracy: {results['accuracy']:.4f}")
        if 'auc' in results:
            print(f"Final AUC: {results['auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())