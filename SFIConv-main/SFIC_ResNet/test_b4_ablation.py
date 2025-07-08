# test_b4_ablation.py - 支持消融实验的测试脚本
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
from network.data import SingleInputDataset, TestDataset
from network.utils import cal_metrics
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class AblationTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        """根据模型配置设置模型"""
        print(f"Setting up model for testing...")
        
        # 首先尝试从检查点加载配置
        model_config = self.load_model_config()
        
        if model_config:
            print(f"Loaded model configuration from checkpoint:")
            print(f"  - AMSFE: {model_config['enable_amsfe']}")
            print(f"  - CLFPF: {model_config['enable_clfpf']}")
            print(f"  - SALGA: {model_config['enable_salga']}")
            
            # 使用保存的配置
            enable_amsfe = model_config['enable_amsfe']
            enable_clfpf = model_config['enable_clfpf']
            enable_salga = model_config['enable_salga']
        else:
            # 使用命令行参数
            print(f"Using command line configuration:")
            print(f"  - AMSFE: {self.args.enable_amsfe}")
            print(f"  - CLFPF: {self.args.enable_clfpf}")
            print(f"  - SALGA: {self.args.enable_salga}")
            
            enable_amsfe = self.args.enable_amsfe
            enable_clfpf = self.args.enable_clfpf
            enable_salga = self.args.enable_salga
        
        # 根据配置创建对应的模型
        if not any([enable_amsfe, enable_clfpf, enable_salga]):
            print("🔵 Creating BASELINE model (standard EfficientNet-B4)")
            self.model = create_model(
                model_type='standard',
                num_classes=self.args.num_classes,
                drop_rate=0.0  # 测试时不使用dropout
            )
        else:
            print("🟡 Creating ENHANCED model with specified modules")
            self.model = create_model(
                model_type='enhanced',
                num_classes=self.args.num_classes,
                drop_rate=0.0,
                enable_amsfe=enable_amsfe,
                enable_clfpf=enable_clfpf,
                enable_salga=enable_salga
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
        
    def load_model_config(self):
        """从检查点或配置文件加载模型配置"""
        try:
            # 尝试从.pth文件加载配置
            if self.args.model_path.endswith('.pth'):
                checkpoint = torch.load(self.args.model_path, map_location='cpu')
                if 'ablation_config' in checkpoint:
                    return checkpoint['ablation_config']
            
            # 尝试从模型目录的配置文件加载
            model_dir = os.path.dirname(self.args.model_path)
            config_file = os.path.join(model_dir, 'ablation_config.json')
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return {
                    'enable_amsfe': config.get('enable_amsfe', False),
                    'enable_clfpf': config.get('enable_clfpf', False),
                    'enable_salga': config.get('enable_salga', False)
                }
            
            return None
            
        except Exception as e:
            print(f"Could not load model config: {e}")
            return None
    
    def load_model(self):
        """加载模型权重"""
        try:
            checkpoint = torch.load(self.args.model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Loaded from checkpoint")
            else:
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
        """设置测试数据加载器"""
        print("Setting up test data...")
        
        if not os.path.exists(self.args.test_txt_path):
            raise FileNotFoundError(f"Test data not found: {self.args.test_txt_path}")
        
        image_size = 224
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        
        if self.args.use_tta:
            self.tta_transforms = self.create_tta_transforms(image_size)
            print(f"TTA enabled with {len(self.tta_transforms)} augmentations")
        
        print(f"Test data loaded: {len(test_dataset)} samples")
    
    def create_tta_transforms(self, image_size):
        """创建TTA变换"""
        base_transform = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        tta_transforms = [
            transforms.Compose(base_transform),
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        ]
        
        return tta_transforms
    
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
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if self.args.num_classes == 2:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
                
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
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                labels = labels.to(self.device)
                
                tta_outputs = []
                
                for tta_transform in self.tta_transforms:
                    augmented_batch = []
                    for i in range(images.size(0)):
                        img_pil = transforms.ToPILImage()(images[i])
                        aug_img = tta_transform(img_pil)
                        augmented_batch.append(aug_img)
                    
                    aug_batch = torch.stack(augmented_batch).to(self.device)
                    outputs = self.model(aug_batch)
                    tta_outputs.append(outputs)
                
                avg_outputs = torch.mean(torch.stack(tta_outputs), dim=0)
                _, predicted = torch.max(avg_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if self.args.num_classes == 2:
                    probs = torch.nn.functional.softmax(avg_outputs, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    current_acc = correct / total if total > 0 else 0
                    print(f"TTA Batch [{batch_idx:4d}/{len(self.test_loader)}] Acc: {current_acc:.2%}")
        
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
                ap_score, auc_score, eer, TPR_2, TPR_3, TPR_4 = cal_metrics(all_labels, all_probs)
                
                results.update({
                    'auc': auc_score,
                    'ap': ap_score,
                    'eer': eer,
                    'tpr_at_fpr_1e-2': TPR_2,
                    'tpr_at_fpr_1e-3': TPR_3,
                    'tpr_at_fpr_1e-4': TPR_4
                })
                
                cm = confusion_matrix(all_labels, all_preds)
                results['confusion_matrix'] = cm.tolist()
                
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
                
                precision = results['precision']
                recall = results['recall']
                results['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                results['labels'] = all_labels
                results['probs'] = all_probs
                results['predictions'] = all_preds
                
            except Exception as e:
                print(f"Error calculating detailed metrics: {e}")
        
        return results
    
    def save_results(self, results, suffix=""):
        """保存测试结果"""
        output_dir = os.path.dirname(self.args.model_path)
        if not output_dir:
            output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_file = os.path.join(output_dir, f'test_results{suffix}.txt')
        with open(result_file, 'w') as f:
            f.write(f"Ablation Test Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model path: {self.args.model_path}\n")
            f.write(f"Test data: {self.args.test_txt_path}\n")
            f.write(f"Use TTA: {self.args.use_tta}\n")
            f.write(f"Enhancement modules: AMSFE={self.args.enable_amsfe}, CLFPF={self.args.enable_clfpf}, SALGA={self.args.enable_salga}\n")
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
        
        json_results = {k: v for k, v in results.items() 
                       if k not in ['labels', 'probs', 'predictions']}
        json_file = os.path.join(output_dir, f'test_results{suffix}.json')
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {result_file}")
        return result_file
    
    def run_test(self):
        """运行测试"""
        print(f"\n🚀 Testing Ablation Model")
        print("="*60)
        
        standard_results = self.test_standard()
        self.save_results(standard_results, "_standard")
        
        if self.args.use_tta:
            print("\nRunning test with TTA...")
            tta_results = self.test_with_tta()
            self.save_results(tta_results, "_tta")
            
            print("\n📊 Results Comparison:")
            print(f"Standard - Acc: {standard_results['accuracy']:.4f}, AUC: {standard_results.get('auc', 0):.4f}")
            print(f"TTA      - Acc: {tta_results['accuracy']:.4f}, AUC: {tta_results.get('auc', 0):.4f}")
            
            return tta_results
        
        return standard_results


def main():
    parser = argparse.ArgumentParser(description='Ablation Model Testing')
    
    # 基本参数
    parser.add_argument('--test_txt_path', type=str, 
                       default='/home/zqc/WDF/test.txt',
                       help='Test data txt path')
    parser.add_argument('--model_path', type=str, 
                       required=True,
                       help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    
    # 消融实验参数（可选，会尝试从模型配置文件自动读取）
    parser.add_argument('--enable_amsfe', action='store_true', default=False,
                       help='Enable AMSFE module (auto-detected if not specified)')
    parser.add_argument('--enable_clfpf', action='store_true', default=False,
                       help='Enable CLFPF module (auto-detected if not specified)')
    parser.add_argument('--enable_salga', action='store_true', default=False,
                       help='Enable SALGA module (auto-detected if not specified)')
    
    # 测试选项
    parser.add_argument('--use_tta', action='store_true', default=False,
                       help='Use test time augmentation')
    
    args = parser.parse_args()
    
    print("🧪 Ablation Model Testing")
    print("="*40)
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_txt_path}")
    print(f"Use TTA: {args.use_tta}")
    print("="*40)
    
    try:
        tester = AblationTester(args)
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