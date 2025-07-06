# paper_visualization_generator.py - 论文级别可视化生成器
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os
from torchvision import transforms
import matplotlib.patches as Rectangle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 导入你的可视化工具
from visualization_tools import DeepfakeVisualizationTools
from network.MainNet import create_model

class PaperVisualizationGenerator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.viz_tools = DeepfakeVisualizationTools(self.model, device)
        
        # 数据路径
        self.real_path = "/home/zqc/FaceForensics++/c23/test/real"
        self.fake_path = "/home/zqc/FaceForensics++/c23/test/fake"
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 论文级别的颜色配置
        self.colors = {
            'amsfe': '#FF6B6B',    # 红色系 - 频域
            'clfpf': '#4ECDC4',    # 青色系 - 金字塔
            'salga': '#45B7D1',    # 蓝色系 - 注意力
            'real': '#2ECC71',     # 绿色 - 真实
            'fake': '#E74C3C'      # 红色 - 伪造
        }
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        model = create_model(
            model_type='enhanced',
            num_classes=2,
            enable_amsfe=True,
            enable_clfpf=True,
            enable_salga=True
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_sample_images(self, num_real=3, num_fake=3):
        """加载代表性样本图像"""
        real_images = []
        fake_images = []
        
        # 加载真实图像
        real_files = sorted([f for f in os.listdir(self.real_path) if f.endswith('.png')])[:num_real]
        for file in real_files:
            img_path = os.path.join(self.real_path, file)
            img = Image.open(img_path).convert('RGB')
            real_images.append((img, file))
        
        # 加载伪造图像
        fake_files = sorted([f for f in os.listdir(self.fake_path) if f.endswith('.png')])[:num_fake]
        for file in fake_files:
            img_path = os.path.join(self.fake_path, file)
            img = Image.open(img_path).convert('RGB')
            fake_images.append((img, file))
        
        return real_images, fake_images
    
    def denormalize_image(self, tensor):
        """反归一化图像用于显示"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        tensor = tensor.clone()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor.permute(1, 2, 0).cpu().numpy()
    
    def generate_architecture_flow_figure(self, save_path='./paper_figures/architecture_flow.png'):
        """生成架构流程图 - Figure 1"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 选择一个代表性样本
        real_images, fake_images = self.load_sample_images(1, 1)
        sample_img, _ = real_images[0]
        sample_tensor = self.transform(sample_img)
        
        # 创建大图布局
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1])
        
        # 获取模型内部特征
        with torch.no_grad():
            _ = self.model(sample_tensor.unsqueeze(0).to(self.device))
        
        original_img = self.denormalize_image(sample_tensor)
        
        # 第一行：输入 -> 骨干网络 -> 三个模块
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_img)
        ax1.set_title('Input Image\n(Real Sample)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, 'EfficientNet-B4\nBackbone\n(Pretrained)', 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # AMSFE模块可视化
        ax3 = fig.add_subplot(gs[0, 2])
        if 'amsfe' in self.viz_tools.features:
            amsfe_feat = torch.mean(self.viz_tools.features['amsfe'], dim=1).squeeze().cpu().numpy()
            amsfe_feat = cv2.resize(amsfe_feat, (224, 224))
            amsfe_feat = (amsfe_feat - amsfe_feat.min()) / (amsfe_feat.max() - amsfe_feat.min())
            im3 = ax3.imshow(amsfe_feat, cmap='Reds', alpha=0.8)
            ax3.imshow(original_img, alpha=0.3)
        ax3.set_title('AMSFE Module\n(Frequency Enhancement)', fontsize=11, fontweight='bold', color=self.colors['amsfe'])
        ax3.axis('off')
        
        # CLFPF模块可视化
        ax4 = fig.add_subplot(gs[0, 3])
        if 'clfpf' in self.viz_tools.features:
            clfpf_feat = torch.mean(self.viz_tools.features['clfpf'], dim=1).squeeze().cpu().numpy()
            clfpf_feat = cv2.resize(clfpf_feat, (224, 224))
            clfpf_feat = (clfpf_feat - clfpf_feat.min()) / (clfpf_feat.max() - clfpf_feat.min())
            im4 = ax4.imshow(clfpf_feat, cmap='Greens', alpha=0.8)
            ax4.imshow(original_img, alpha=0.3)
        ax4.set_title('CLFPF Module\n(Pyramid Fusion)', fontsize=11, fontweight='bold', color=self.colors['clfpf'])
        ax4.axis('off')
        
        # SALGA模块可视化
        ax5 = fig.add_subplot(gs[0, 4])
        if 'salga' in self.viz_tools.features:
            salga_feat = torch.mean(self.viz_tools.features['salga'], dim=1).squeeze().cpu().numpy()
            salga_feat = cv2.resize(salga_feat, (224, 224))
            salga_feat = (salga_feat - salga_feat.min()) / (salga_feat.max() - salga_feat.min())
            im5 = ax5.imshow(salga_feat, cmap='Blues', alpha=0.8)
            ax5.imshow(original_img, alpha=0.3)
        ax5.set_title('SALGA Module\n(Semantic Attention)', fontsize=11, fontweight='bold', color=self.colors['salga'])
        ax5.axis('off')
        
        # 输出
        ax6 = fig.add_subplot(gs[0, 5])
        with torch.no_grad():
            output = self.model(sample_tensor.unsqueeze(0).to(self.device))
            prob = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(output, dim=1).item()
        
        ax6.text(0.5, 0.5, f'Prediction:\n{"Real" if pred_class==0 else "Fake"}\nConf: {prob[pred_class]:.3f}', 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor=self.colors['real'] if pred_class==0 else self.colors['fake'],
                         alpha=0.7))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        # 第二行：特征图可视化
        module_names = ['amsfe', 'clfpf', 'salga']
        for i, module in enumerate(module_names):
            if module in self.viz_tools.features:
                for j in range(2):  # 显示每个模块的2个特征通道
                    ax = fig.add_subplot(gs[1, i*2+j])
                    features = self.viz_tools.features[module].squeeze().cpu().numpy()
                    if j < features.shape[0]:
                        feat_map = features[j]
                        if feat_map.max() > feat_map.min():
                            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
                        ax.imshow(feat_map, cmap='viridis')
                    ax.set_title(f'{module.upper()}\nCh.{j+1}', fontsize=10)
                    ax.axis('off')
        
        # 第三行：Grad-CAM和统计
        ax_cam = fig.add_subplot(gs[2, :2])
        cam = self.viz_tools.grad_cam_with_batch(sample_tensor, target_layer='amsfe')
        if cam is not None:
            im_cam = ax_cam.imshow(cam, cmap='jet', alpha=0.7)
            ax_cam.imshow(original_img, alpha=0.4)
            plt.colorbar(im_cam, ax=ax_cam, fraction=0.046)
        ax_cam.set_title('Grad-CAM (AMSFE Focus)', fontsize=12, fontweight='bold')
        ax_cam.axis('off')
        
        # 模块激活统计
        ax_stats = fig.add_subplot(gs[2, 2:4])
        module_activations = []
        module_labels = []
        for name in ['amsfe', 'clfpf', 'salga']:
            if name in self.viz_tools.features:
                activation = torch.mean(self.viz_tools.features[name]).item()
                module_activations.append(activation)
                module_labels.append(name.upper())
        
        bars = ax_stats.bar(module_labels, module_activations, 
                           color=[self.colors['amsfe'], self.colors['clfpf'], self.colors['salga']])
        ax_stats.set_title('Module Activation Strength', fontsize=12, fontweight='bold')
        ax_stats.set_ylabel('Average Activation')
        for bar, val in zip(bars, module_activations):
            ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加箭头连接
        self.add_flow_arrows(fig, gs)
        
        plt.suptitle('Enhanced EfficientNet-B4 Architecture Flow for Deepfake Detection', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Architecture flow figure saved to: {save_path}")
    
    def add_flow_arrows(self, fig, gs):
        """添加流程箭头"""
        # 这里可以添加箭头来显示数据流
        pass
    
    def generate_real_vs_fake_comparison(self, save_path='./paper_figures/real_vs_fake_comparison.png'):
        """生成真实vs伪造对比图 - Figure 2"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        real_images, fake_images = self.load_sample_images(2, 2)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for i in range(2):  # 2个样本
            # 真实图像
            real_img, real_file = real_images[i]
            real_tensor = self.transform(real_img)
            
            # 伪造图像
            fake_img, fake_file = fake_images[i]
            fake_tensor = self.transform(fake_img)
            
            # 处理真实图像
            with torch.no_grad():
                _ = self.model(real_tensor.unsqueeze(0).to(self.device))
                real_features = {k: v.clone() for k, v in self.viz_tools.features.items()}
            
            # 显示原图
            axes[i*2, 0].imshow(self.denormalize_image(real_tensor))
            axes[i*2, 0].set_title(f'Real Image {i+1}', fontweight='bold', color=self.colors['real'])
            axes[i*2, 0].axis('off')
            
            # 显示各模块注意力
            for j, module in enumerate(['amsfe', 'clfpf', 'salga']):
                if module in real_features:
                    feat = torch.mean(real_features[module], dim=1).squeeze().cpu().numpy()
                    feat = cv2.resize(feat, (224, 224))
                    feat = (feat - feat.min()) / (feat.max() - feat.min())
                    
                    axes[i*2, j+1].imshow(feat, cmap='jet', alpha=0.8)
                    axes[i*2, j+1].imshow(self.denormalize_image(real_tensor), alpha=0.3)
                    axes[i*2, j+1].set_title(f'{module.upper()}\n(Real)', fontsize=10)
                    axes[i*2, j+1].axis('off')
            
            # 处理伪造图像
            with torch.no_grad():
                _ = self.model(fake_tensor.unsqueeze(0).to(self.device))
                fake_features = {k: v.clone() for k, v in self.viz_tools.features.items()}
            
            # 显示原图
            axes[i*2+1, 0].imshow(self.denormalize_image(fake_tensor))
            axes[i*2+1, 0].set_title(f'Fake Image {i+1}', fontweight='bold', color=self.colors['fake'])
            axes[i*2+1, 0].axis('off')
            
            # 显示各模块注意力
            for j, module in enumerate(['amsfe', 'clfpf', 'salga']):
                if module in fake_features:
                    feat = torch.mean(fake_features[module], dim=1).squeeze().cpu().numpy()
                    feat = cv2.resize(feat, (224, 224))
                    feat = (feat - feat.min()) / (feat.max() - feat.min())
                    
                    axes[i*2+1, j+1].imshow(feat, cmap='jet', alpha=0.8)
                    axes[i*2+1, j+1].imshow(self.denormalize_image(fake_tensor), alpha=0.3)
                    axes[i*2+1, j+1].set_title(f'{module.upper()}\n(Fake)', fontsize=10)
                    axes[i*2+1, j+1].axis('off')
        
        plt.suptitle('Real vs Fake Image Analysis: Module-wise Attention Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Real vs Fake comparison saved to: {save_path}")
    
    def generate_module_effectiveness_analysis(self, save_path='./paper_figures/module_effectiveness.png'):
        """生成模块有效性分析 - Figure 3"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        real_images, fake_images = self.load_sample_images(5, 5)
        
        # 统计各模块对真实和伪造图像的激活差异
        real_activations = {'amsfe': [], 'clfpf': [], 'salga': []}
        fake_activations = {'amsfe': [], 'clfpf': [], 'salga': []}
        
        # 处理真实图像
        for real_img, _ in real_images:
            real_tensor = self.transform(real_img)
            with torch.no_grad():
                _ = self.model(real_tensor.unsqueeze(0).to(self.device))
                for module in ['amsfe', 'clfpf', 'salga']:
                    if module in self.viz_tools.features:
                        activation = torch.mean(self.viz_tools.features[module]).item()
                        real_activations[module].append(activation)
        
        # 处理伪造图像
        for fake_img, _ in fake_images:
            fake_tensor = self.transform(fake_img)
            with torch.no_grad():
                _ = self.model(fake_tensor.unsqueeze(0).to(self.device))
                for module in ['amsfe', 'clfpf', 'salga']:
                    if module in self.viz_tools.features:
                        activation = torch.mean(self.viz_tools.features[module]).item()
                        fake_activations[module].append(activation)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 上排：各模块激活对比
        modules = ['amsfe', 'clfpf', 'salga']
        module_titles = ['AMSFE (Frequency)', 'CLFPF (Pyramid)', 'SALGA (Attention)']
        
        for i, (module, title) in enumerate(zip(modules, module_titles)):
            real_vals = real_activations[module]
            fake_vals = fake_activations[module]
            
            # 箱线图
            axes[0, i].boxplot([real_vals, fake_vals], labels=['Real', 'Fake'],
                              patch_artist=True,
                              boxprops=dict(facecolor=self.colors['real'], alpha=0.7),
                              medianprops=dict(color='black', linewidth=2))
            axes[0, i].set_title(f'{title}\nActivation Distribution', fontweight='bold')
            axes[0, i].set_ylabel('Activation Value')
            axes[0, i].grid(True, alpha=0.3)
        
        # 下排：特征分布热力图
        sample_real, _ = real_images[0]
        sample_fake, _ = fake_images[0]
        
        real_tensor = self.transform(sample_real)
        fake_tensor = self.transform(sample_fake)
        
        # 获取特征
        with torch.no_grad():
            _ = self.model(real_tensor.unsqueeze(0).to(self.device))
            real_features = {k: v.clone() for k, v in self.viz_tools.features.items()}
            
            _ = self.model(fake_tensor.unsqueeze(0).to(self.device))
            fake_features = {k: v.clone() for k, v in self.viz_tools.features.items()}
        
        for i, module in enumerate(modules):
            if module in real_features and module in fake_features:
                # 计算特征差异
                real_feat = torch.mean(real_features[module], dim=1).squeeze().cpu().numpy()
                fake_feat = torch.mean(fake_features[module], dim=1).squeeze().cpu().numpy()
                
                # 调整大小并计算差异
                real_feat = cv2.resize(real_feat, (56, 56))
                fake_feat = cv2.resize(fake_feat, (56, 56))
                diff = np.abs(real_feat - fake_feat)
                
                im = axes[1, i].imshow(diff, cmap='hot', interpolation='bilinear')
                axes[1, i].set_title(f'{module.upper()}\nReal-Fake Difference', fontweight='bold')
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046)
        
        plt.suptitle('Module Effectiveness Analysis: Discriminative Power Evaluation', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Module effectiveness analysis saved to: {save_path}")
    
    def generate_gradcam_comparison(self, save_path='./paper_figures/gradcam_comparison.png'):
        """生成Grad-CAM对比分析 - Figure 4"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        real_images, fake_images = self.load_sample_images(2, 2)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # 对每个样本生成Grad-CAM
        samples = [(real_images[0], 'Real 1'), (fake_images[0], 'Fake 1'),
                  (real_images[1], 'Real 2'), (fake_images[1], 'Fake 2')]
        
        for col, ((img, filename), label) in enumerate(samples):
            img_tensor = self.transform(img)
            original_img = self.denormalize_image(img_tensor)
            
            # 原始图像
            axes[0, col].imshow(original_img)
            axes[0, col].set_title(f'{label}\n({filename})', fontweight='bold',
                                  color=self.colors['real'] if 'Real' in label else self.colors['fake'])
            axes[0, col].axis('off')
            
            # 各模块的Grad-CAM
            for row, module in enumerate(['amsfe', 'clfpf'], 1):
                cam = self.viz_tools.grad_cam_with_batch(img_tensor, target_layer=module)
                if cam is not None:
                    im = axes[row, col].imshow(cam, cmap='jet', alpha=0.7)
                    axes[row, col].imshow(original_img, alpha=0.4)
                    axes[row, col].set_title(f'{module.upper()} Grad-CAM', fontsize=10)
                    axes[row, col].axis('off')
                    
                    # 只在第一列添加colorbar
                    if col == 0:
                        plt.colorbar(im, ax=axes[row, col], fraction=0.046)
        
        plt.suptitle('Grad-CAM Analysis: Model Decision Focus Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Grad-CAM comparison saved to: {save_path}")
    
    def generate_all_paper_figures(self, output_dir='./paper_figures'):
        """生成所有论文图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎨 Generating comprehensive paper visualizations...")
        print("="*60)
        
        # Figure 1: 架构流程图
        print("📊 Generating Figure 1: Architecture Flow...")
        self.generate_architecture_flow_figure(
            os.path.join(output_dir, 'figure1_architecture_flow.png')
        )
        
        # Figure 2: 真实vs伪造对比
        print("🔍 Generating Figure 2: Real vs Fake Comparison...")
        self.generate_real_vs_fake_comparison(
            os.path.join(output_dir, 'figure2_real_vs_fake.png')
        )
        
        # Figure 3: 模块有效性分析
        print("📈 Generating Figure 3: Module Effectiveness...")
        self.generate_module_effectiveness_analysis(
            os.path.join(output_dir, 'figure3_module_effectiveness.png')
        )
        
        # Figure 4: Grad-CAM对比
        print("🎯 Generating Figure 4: Grad-CAM Comparison...")
        self.generate_gradcam_comparison(
            os.path.join(output_dir, 'figure4_gradcam_comparison.png')
        )
        
        print("="*60)
        print(f"✅ All paper figures generated successfully!")
        print(f"📁 Saved to: {output_dir}")
        print("\n📋 Generated figures:")
        print("  - figure1_architecture_flow.png (流程架构图)")
        print("  - figure2_real_vs_fake.png (真实vs伪造对比)")
        print("  - figure3_module_effectiveness.png (模块有效性分析)")
        print("  - figure4_gradcam_comparison.png (Grad-CAM对比)")

# 使用示例
def main():
    """主函数"""
    print("🚀 Starting Paper Visualization Generation")
    print("="*50)
    
    # 模型路径
    model_path = './output/efficientnet-b4-enhanced/best.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure you have a trained model at the specified path.")
        return
    
    try:
        # 创建可视化生成器
        generator = PaperVisualizationGenerator(model_path)
        
        # 生成所有论文图表
        generator.generate_all_paper_figures()
        
        print("\n🎉 Paper visualization generation completed!")
        print("💡 These figures are ready for inclusion in your research paper.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()