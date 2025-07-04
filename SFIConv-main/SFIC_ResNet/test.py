'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from network.data import TestDataset
from network.transform import Data_Transforms
from network.MainNet import MainNet, EnhancedMainNet  # 添加EnhancedMainNet
from network.plot_roc import plot_ROC
from network.utils import cal_metrics  # 添加更详细的指标计算
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    args = parse.parse_args()
    test_txt_path = args.test_txt_path
    batch_size = args.batch_size
    model_path = args.model_path
    num_classes = args.num_classes
    use_enhanced = args.use_enhanced
	
    torch.backends.cudnn.benchmark = True
	
    # -----create test data----- #
    test_data = TestDataset(txt_path=test_txt_path, test_transform=Data_Transforms['test'])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Test data loaded: {len(test_data)} samples")
    
    # -----create model----- #
    if use_enhanced:
        print("Using Enhanced MainNet for testing")
        model = EnhancedMainNet(
            num_classes=num_classes,
            enable_amsfe=args.enable_amsfe,
            enable_clfpf=args.enable_clfpf,
            enable_salga=args.enable_salga
        )
    else:
        print("Using Original MainNet for testing")
        model = MainNet(num_classes)
    
    # 加载模型权重
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"Model path not found: {model_path}")
        return
	
    if isinstance(model, nn.DataParallel):
        model = model.module
	
    model = model.cuda()
    model.eval()
    
    print("Starting testing...")

    correct_test = 0.0
    total_test_samples = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img_rgb, labels_test = data

            img_rgb = img_rgb.cuda()
            labels_test = labels_test.cuda()
            
            # feed data
            pre_test = model(img_rgb)
            
            # prediction
            _, pred = torch.max(pre_test.data, 1)
            
            # the number of all testing sample
            total_test_samples += labels_test.size(0)
            
            # the correct number of prediction
            correct_test += (pred == labels_test).squeeze().sum().cpu().numpy()
            
            # compute ROC
            pre_test_abs = torch.nn.functional.softmax(pre_test, dim=1)
            pred_abs_temp = torch.zeros(pre_test_abs.size()[0])
            for m in range(pre_test_abs.size()[0]):
                pred_abs_temp[m] = pre_test_abs[m][1]

            label_test_list.extend(labels_test.detach().cpu().numpy())
            predict_test_list.extend(pred_abs_temp.detach().cpu().numpy())
            
            # 显示进度
            if i % 100 == 0:
                current_acc = correct_test / total_test_samples
                print(f"Batch [{i:4d}/{len(test_loader)}] Current Acc: {current_acc:.2%}")
        
    # 最终准确率
    final_acc = correct_test / total_test_samples
    print(f"\nTesting Acc: {final_acc:.2%}")
    
    # 计算更详细的指标（如果是二分类）
    if num_classes == 2:
        try:
            ap_score, auc_score, eer, TPR_2, TPR_3, TPR_4 = cal_metrics(label_test_list, predict_test_list)
            print(f"Detailed Results:")
            print(f"  Accuracy: {final_acc:.4f}")
            print(f"  AUC:      {auc_score:.4f}")
            print(f"  AP:       {ap_score:.4f}")
            print(f"  EER:      {eer:.4f}")
            print(f"  TPR@FPR=1e-2: {TPR_2:.4f}")
            print(f"  TPR@FPR=1e-3: {TPR_3:.4f}")
            print(f"  TPR@FPR=1e-4: {TPR_4:.4f}")
        except Exception as e:
            print(f"Could not calculate detailed metrics: {e}")

    # ROC curve
    try:
        plot_ROC(label_test_list, predict_test_list)
        print("ROC curve saved to ./output/roc.png")
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")

if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--test_txt_path', '-tp', type=str, default='/home/zqc/FaceForensics++/c23/test.txt')
    parse.add_argument('--model_path', '-mp', type=str, default='./output/enhanced-test/best.pkl')
    parse.add_argument('--num_classes', '-nc', type=int, default=2)
    
    # 新增参数：支持增强模型测试
    parse.add_argument('--use_enhanced', action='store_true', help='Use enhanced model for testing')
    parse.add_argument('--enable_amsfe', action='store_true', help='Enable AMSFE module')
    parse.add_argument('--enable_clfpf', action='store_true', help='Enable CLFPF module')
    parse.add_argument('--enable_salga', action='store_true', help='Enable SALGA module')
    
    label_test_list = []
    predict_test_list = []
    
    main()