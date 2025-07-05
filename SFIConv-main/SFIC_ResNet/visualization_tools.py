# visualization_tools.py - 深度伪造检测可视化工具集
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.patches as patches
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class DeepfakeVisualizationTools:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 注册钩子函数用于特征提取
        self.features = {}
        self.gradients = {}
        self.register_hooks()
        
    def register_hooks(self):
        """注册钩子函数用于特征和梯度提取"""
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        def get_gradients(name):
            def hook(model, input, output):
                def backward_hook(grad):
                    self.gradients[name] = grad
                output.register_hook(backward_hook)
            return hook
        
        # 为不同层注册钩子
        if hasattr(self.model, 'backbone'):
            # EfficientNet backbone layers
            if hasattr(self.model.backbone, 'features'):
                for i, layer in enumerate(self.model.backbone.features):
                    layer.register_forward_hook(get_features(f'backbone_layer_{i}'))
                    layer.register_forward_hook(get_gradients(f'backbone_layer_{i}'))
        
        # 为增强模块注册钩子
        if hasattr(self.model, 'amsfe'):
            self.model.amsfe.register_forward_hook(get_features('amsfe'))
            self.model.amsfe.register_forward_hook(get_gradients('amsfe'))
        
        if hasattr(self.model, 'clfpf'):
            self.model.clfpf.register_forward_hook(get_features('clfpf'))
            self.model.clfpf.register_forward_hook(get_gradients('clfpf'))
        
        if hasattr(self.model, 'salga'):
            self.model.salga.register_forward_hook(get_features('salga'))
            self.model.salga.register_forward_hook(get_gradients('salga'))

    def grad_cam(self, image, target_layer='amsfe', class_idx=None):
        """生成Grad-CAM热力图"""
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 前向传播
        output = self.model(image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # 获取特征和梯度
        if target_layer in self.features and target_layer in self.gradients:
            features = self.features[target_layer]
            gradients = self.gradients[target_layer]
            
            # 计算权重
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # 生成CAM
            cam = torch.sum(weights * features, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            return cam
        else:
            print(f"Layer {target_layer} not found in registered hooks")
            return None

    def visualize_attention_maps(self, image, save_path=None):
        """可视化不同模块的注意力图"""
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # AMSFE注意力图
        if 'amsfe' in self.features:
            amsfe_features = self.features['amsfe']
            amsfe_attention = torch.mean(amsfe_features, dim=1).squeeze().cpu().numpy()
            amsfe_attention = cv2.resize(amsfe_attention, (224, 224))
            amsfe_attention = (amsfe_attention - amsfe_attention.min()) / (amsfe_attention.max() - amsfe_attention.min())
            
            im1 = axes[0, 1].imshow(amsfe_attention, cmap='jet', alpha=0.7)
            axes[0, 1].imshow(img_np, alpha=0.3)
            axes[0, 1].set_title('AMSFE Attention')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # CLFPF注意力图
        if 'clfpf' in self.features:
            clfpf_features = self.features['clfpf']
            clfpf_attention = torch.mean(clfpf_features, dim=1).squeeze().cpu().numpy()
            clfpf_attention = cv2.resize(clfpf_attention, (224, 224))
            clfpf_attention = (clfpf_attention - clfpf_attention.min()) / (clfpf_attention.max() - clfpf_attention.min())
            
            im2 = axes[0, 2].imshow(clfpf_attention, cmap='jet', alpha=0.7)
            axes[0, 2].imshow(img_np, alpha=0.3)
            axes[0, 2].set_title('CLFPF Attention')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # SALGA注意力图
        if 'salga' in self.features:
            salga_features = self.features['salga']
            salga_attention = torch.mean(salga_features, dim=1).squeeze().cpu().numpy()
            salga_attention = cv2.resize(salga_attention, (224, 224))
            salga_attention = (salga_attention - salga_attention.min()) / (salga_attention.max() - salga_attention.min())
            
            im3 = axes[1, 0].imshow(salga_attention, cmap='jet', alpha=0.7)
            axes[1, 0].imshow(img_np, alpha=0.3)
            axes[1, 0].set_title('SALGA Attention')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Grad-CAM (整体)
        cam = self.grad_cam(image, target_layer='amsfe')
        if cam is not None:
            im4 = axes[1, 1].imshow(cam, cmap='jet', alpha=0.7)
            axes[1, 1].imshow(img_np, alpha=0.3)
            axes[1, 1].set_title('Grad-CAM (AMSFE)')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        # 特征差异图
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_frequency_analysis(self, real_images, fake_images, save_path=None):
        """频域分析可视化"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i, (real_img, fake_img) in enumerate(zip(real_images[:2], fake_images[:2])):
            # 转换为numpy
            real_np = real_img.permute(1, 2, 0).cpu().numpy()
            fake_np = fake_img.permute(1, 2, 0).cpu().numpy()
            
            # 去归一化
            real_np = (real_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            fake_np = (fake_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            
            # 转换为灰度图
            real_gray = cv2.cvtColor((real_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            fake_gray = cv2.cvtColor((fake_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # FFT变换
            real_fft = np.fft.fft2(real_gray)
            fake_fft = np.fft.fft2(fake_gray)
            
            real_fft_shift = np.fft.fftshift(real_fft)
            fake_fft_shift = np.fft.fftshift(fake_fft)
            
            real_magnitude = np.log(np.abs(real_fft_shift) + 1)
            fake_magnitude = np.log(np.abs(fake_fft_shift) + 1)
            
            # 可视化
            axes[i, 0].imshow(real_np)
            axes[i, 0].set_title(f'Real Image {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(fake_np)
            axes[i, 1].set_title(f'Fake Image {i+1}')
            axes[i, 1].axis('off')
            
            im1 = axes[i, 2].imshow(real_magnitude, cmap='hot')
            axes[i, 2].set_title(f'Real FFT {i+1}')
            axes[i, 2].axis('off')
            plt.colorbar(im1, ax=axes[i, 2], fraction=0.046)
            
            im2 = axes[i, 3].imshow(fake_magnitude, cmap='hot')
            axes[i, 3].set_title(f'Fake FFT {i+1}')
            axes[i, 3].axis('off')
            plt.colorbar(im2, ax=axes[i, 3], fraction=0.046)
        
        # 频域差异分析
        diff_magnitude = np.abs(real_magnitude - fake_magnitude)
        im3 = axes[2, 0].imshow(diff_magnitude, cmap='coolwarm')
        axes[2, 0].set_title('FFT Difference')
        axes[2, 0].axis('off')
        plt.colorbar(im3, ax=axes[2, 0], fraction=0.046)
        
        # 频域统计
        axes[2, 1].hist(real_magnitude.flatten(), bins=50, alpha=0.5, label='Real', color='blue')
        axes[2, 1].hist(fake_magnitude.flatten(), bins=50, alpha=0.5, label='Fake', color='red')
        axes[2, 1].set_title('FFT Magnitude Distribution')
        axes[2, 1].legend()
        
        # 隐藏剩余子图
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_feature_maps(self, image, layer_names=['amsfe', 'clfpf', 'salga'], save_path=None):
        """可视化不同层的特征图"""
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))
        
        num_layers = len(layer_names)
        fig, axes = plt.subplots(num_layers, 8, figsize=(20, 3*num_layers))
        
        for i, layer_name in enumerate(layer_names):
            if layer_name in self.features:
                features = self.features[layer_name].squeeze().cpu().numpy()
                
                # 选择前8个通道进行可视化
                num_channels = min(8, features.shape[0])
                
                for j in range(num_channels):
                    feature_map = features[j]
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                    
                    axes[i, j].imshow(feature_map, cmap='viridis')
                    axes[i, j].set_title(f'{layer_name.upper()} Ch.{j+1}')
                    axes[i, j].axis('off')
                
                # 隐藏多余的子图
                for j in range(num_channels, 8):
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_roc_curves(self, y_true_dict, y_scores_dict, save_path=None):
        """可视化多个模型的ROC曲线对比"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, y_scores) in enumerate(y_scores_dict.items()):
            y_true = y_true_dict[model_name]
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_tsne_features(self, features_dict, labels_dict, save_path=None):
        """使用t-SNE可视化特征分布"""
        fig, axes = plt.subplots(1, len(features_dict), figsize=(5*len(features_dict), 5))
        
        if len(features_dict) == 1:
            axes = [axes]
        
        for i, (model_name, features) in enumerate(features_dict.items()):
            labels = labels_dict[model_name]
            
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features)
            
            # 绘制散点图
            scatter = axes[i].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=labels, cmap='RdYlBu', alpha=0.7, s=20)
            axes[i].set_title(f't-SNE: {model_name}', fontsize=12)
            axes[i].set_xlabel('t-SNE 1')
            axes[i].set_ylabel('t-SNE 2')
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=axes[i])
            cbar.set_label('Real (0) / Fake (1)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_confusion_matrices(self, y_true_dict, y_pred_dict, save_path=None):
        """可视化混淆矩阵对比"""
        num_models = len(y_true_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 4))
        
        if num_models == 1:
            axes = [axes]
        
        for i, (model_name, y_true) in enumerate(y_true_dict.items()):
            y_pred = y_pred_dict[model_name]
            cm = confusion_matrix(y_true, y_pred)
            
            # 归一化
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 绘制热力图
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                       ax=axes[i])
            axes[i].set_title(f'Confusion Matrix: {model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_detection_examples(self, test_loader, num_examples=8, save_path=None):
        """可视化检测结果示例"""
        fig, axes = plt.subplots(2, num_examples, figsize=(2*num_examples, 6))
        
        correct_count = 0
        wrong_count = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
                
                for i, (image, label, pred_class, pred_prob) in enumerate(
                    zip(images, labels, predicted_classes, predictions)):
                    
                    if correct_count < num_examples//2 and pred_class == label:
                        # 正确预测的例子
                        img_np = image.permute(1, 2, 0).cpu().numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                                 np.array([0.485, 0.456, 0.406])).clip(0, 1)
                        
                        axes[0, correct_count].imshow(img_np)
                        axes[0, correct_count].set_title(
                            f'✓ True: {"Fake" if label==1 else "Real"}\n'
                            f'Pred: {"Fake" if pred_class==1 else "Real"} '
                            f'({pred_prob[pred_class]:.3f})', 
                            color='green', fontsize=10)
                        axes[0, correct_count].axis('off')
                        correct_count += 1
                    
                    elif wrong_count < num_examples//2 and pred_class != label:
                        # 错误预测的例子
                        img_np = image.permute(1, 2, 0).cpu().numpy()
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                                 np.array([0.485, 0.456, 0.406])).clip(0, 1)
                        
                        axes[1, wrong_count].imshow(img_np)
                        axes[1, wrong_count].set_title(
                            f'✗ True: {"Fake" if label==1 else "Real"}\n'
                            f'Pred: {"Fake" if pred_class==1 else "Real"} '
                            f'({pred_prob[pred_class]:.3f})', 
                            color='red', fontsize=10)
                        axes[1, wrong_count].axis('off')
                        wrong_count += 1
                    
                    if correct_count >= num_examples//2 and wrong_count >= num_examples//2:
                        break
                
                if correct_count >= num_examples//2 and wrong_count >= num_examples//2:
                    break
        
        # 隐藏多余的子图
        for j in range(correct_count, num_examples):
            axes[0, j].axis('off')
        for j in range(wrong_count, num_examples):
            axes[1, j].axis('off')
        
        axes[0, 0].set_ylabel('Correct Predictions', fontsize=12, rotation=90)
        axes[1, 0].set_ylabel('Wrong Predictions', fontsize=12, rotation=90)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_comprehensive_visualization_report(self, test_loader, save_dir='./visualizations'):
        """生成完整的可视化报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("🎨 Generating comprehensive visualization report...")
        
        # 1. 检测结果示例
        print("📸 Creating detection examples...")
        self.visualize_detection_examples(
            test_loader, 
            save_path=os.path.join(save_dir, 'detection_examples.png')
        )
        
        # 2. 注意力可视化
        print("🔍 Creating attention visualizations...")
        sample_batch = next(iter(test_loader))
        sample_image = sample_batch[0][0]  # 取第一张图像
        
        self.visualize_attention_maps(
            sample_image,
            save_path=os.path.join(save_dir, 'attention_maps.png')
        )
        
        # 3. 特征图可视化
        print("🗺️ Creating feature maps...")
        self.visualize_feature_maps(
            sample_image,
            save_path=os.path.join(save_dir, 'feature_maps.png')
        )
        
        print(f"✅ Visualization report saved to {save_dir}")

# 使用示例代码
def main():
    """可视化工具使用示例"""
    # 假设你已经有了训练好的模型和测试数据
    from network.MainNet import create_model
    from torch.utils.data import DataLoader
    
    # 加载模型
    model = create_model(model_type='enhanced', enable_amsfe=True, 
                        enable_clfpf=True, enable_salga=True)
    model.load_state_dict(torch.load('best_model.pkl'))
    
    # 创建可视化工具
    viz_tools = DeepfakeVisualizationTools(model)
    
    # 加载测试数据
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 生成完整的可视化报告
    # viz_tools.create_comprehensive_visualization_report(test_loader)
    
    print("Visualization tools ready to use!")

if __name__ == "__main__":
    main()