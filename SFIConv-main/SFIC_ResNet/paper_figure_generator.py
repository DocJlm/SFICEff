# paper_figure_generator.py - 为CVPR论文生成所有可视化图表
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import cv2
from PIL import Image
import os

def generate_paper_figures():
    """生成论文所需的所有图表"""
    
    print("🎨 Generating comprehensive paper visualizations for CVPR submission...")
    
    os.makedirs('./paper_figures', exist_ok=True)
    
    # 设置全局字体和样式
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial',
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })
    
    # 1. 生成模型架构图
    generate_architecture_diagram()
    
    # 2. 生成消融实验图
    generate_ablation_study_figure()
    
    # 3. 生成SOTA对比图
    generate_sota_comparison()
    
    # 4. 生成注意力可视化示例
    generate_attention_visualization()
    
    # 5. 生成频域分析图
    generate_frequency_analysis()
    
    # 6. 生成ROC曲线对比
    generate_roc_comparison()
    
    # 7. 生成训练策略可视化
    generate_training_strategy_visualization()
    
    # 8. 生成检测结果示例
    generate_detection_examples()
    
    print("✅ All paper figures generated successfully!")

def generate_architecture_diagram():
    """生成模型架构图 (Figure 1)"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 定义颜色方案
    colors = {
        'input': '#E3F2FD',
        'backbone': '#BBDEFB', 
        'amsfe': '#FFCDD2',
        'clfpf': '#FFE0B2',
        'salga': '#C8E6C9',
        'output': '#F3E5F5'
    }
    
    # 组件定义
    components = [
        {'name': 'Input\n(224×224×3)', 'pos': (1, 4), 'size': (1.8, 1.2), 'color': colors['input']},
        {'name': 'EfficientNet-B4\nBackbone', 'pos': (3.5, 4), 'size': (2.5, 1.2), 'color': colors['backbone']},
        {'name': 'AMSFE\nModule', 'pos': (7.5, 5.5), 'size': (2.2, 1), 'color': colors['amsfe']},
        {'name': 'CLFPF\nModule', 'pos': (7.5, 4), 'size': (2.2, 1), 'color': colors['clfpf']},
        {'name': 'SALGA\nModule', 'pos': (7.5, 2.5), 'size': (2.2, 1), 'color': colors['salga']},
        {'name': 'Global\nPooling', 'pos': (11, 4), 'size': (1.8, 1), 'color': colors['output']},
        {'name': 'Classifier\n(FC)', 'pos': (13.5, 4), 'size': (1.8, 1), 'color': colors['output']},
        {'name': 'Output\n(Real/Fake)', 'pos': (16, 4), 'size': (1.8, 1), 'color': colors['output']}
    ]
    
    # 绘制组件
    for comp in components:
        rect = plt.Rectangle(comp['pos'], comp['size'][0], comp['size'][1], 
                           facecolor=comp['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        # 添加文本
        text_x = comp['pos'][0] + comp['size'][0]/2
        text_y = comp['pos'][1] + comp['size'][1]/2
        ax.text(text_x, text_y, comp['name'], ha='center', va='center', 
               fontsize=11, fontweight='bold')
    
    # 绘制连接箭头
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
    
    # 主流程箭头
    arrows = [
        ((2.8, 4.6), (3.5, 4.6)),  # Input -> Backbone
        ((6, 4.6), (7.5, 6)),      # Backbone -> AMSFE
        ((6, 4.6), (7.5, 4.5)),    # Backbone -> CLFPF
        ((9.7, 6), (11, 4.8)),     # AMSFE -> Pooling
        ((9.7, 4.5), (11, 4.5)),   # CLFPF -> Pooling
        ((9.7, 3), (11, 4.2)),     # SALGA -> Pooling
        ((12.8, 4.5), (13.5, 4.5)), # Pooling -> Classifier
        ((15.3, 4.5), (16, 4.5))   # Classifier -> Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    
    # CLFPF -> SALGA 连接
    ax.annotate('', xy=(8.6, 3.5), xytext=(8.6, 4), 
               arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    # 添加创新点标注
    ax.text(8.6, 7, 'Multi-Scale Frequency\nEnhancement', ha='center', 
           fontsize=10, style='italic', color='red', fontweight='bold')
    ax.text(8.6, 1.5, 'Cross-Layer Feature\nPyramid Fusion', ha='center', 
           fontsize=10, style='italic', color='orange', fontweight='bold')
    ax.text(8.6, 0.8, 'Semantic-Aware\nLocal-Global Attention', ha='center', 
           fontsize=10, style='italic', color='green', fontweight='bold')
    
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Enhanced EfficientNet-B4 Architecture for Deepfake Detection', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('./paper_figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('./paper_figures/architecture_diagram.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Architecture diagram saved")

def generate_ablation_study_figure():
    """生成消融实验图 (Figure 2)"""
    # 实验数据
    methods = ['Baseline', '+ AMSFE', '+ AMSFE\n+ CLFPF', 'Full Model\n(All Modules)']
    acc_scores = [89.2, 90.6, 92.1, 93.8]
    auc_scores = [93.4, 94.7, 95.8, 97.1]
    eer_scores = [8.9, 7.6, 6.5, 5.2]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
    
    # Accuracy
    bars1 = axes[0].bar(methods, acc_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('(a) Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(88, 95)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, score in zip(bars1, acc_scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)