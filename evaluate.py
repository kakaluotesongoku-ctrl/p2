import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.models import get_model

# 结果可视化与评估工具

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14)) # 增大画布尺寸
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_loss_curve(log_file_path):
    """
    [TODO任务]
    读取训练日志 (log.txt 或类似文件)，绘制 Loss 和 Accuracy 变化曲线
    """
    # TODO: 实现读取日志并画图的功能
    pass

def analyze_bad_cases(model, dataloader):
    """
    [TODO任务]
    找出模型预测错误的样本，保存下来用于报告分析
    """
    # TODO: 实现错误样本分析
    pass

def tta_predict(model, device, inputs):
    """
    Test Time Augmentation (TTA)
    对输入图像进行多次增强预测并取平均
    """
    # 1. 原图
    outputs = [model(inputs)]
    
    # 2. 水平翻转
    outputs.append(model(torch.flip(inputs, [3])))
    
    # 3. 垂直翻转 (可选，植物叶片垂直翻转也合理)
    # outputs.append(model(torch.flip(inputs, [2])))
    
    # 取平均
    outputs = torch.stack(outputs).mean(0)
    return outputs

def evaluate(model_path, data_dir, model_name='resnet50', batch_size=32, use_tta=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model: {model_path} on {device}")
    
    # 加载数据 (只用 Test 集)
    _, _, test_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(classes)
    
    # 加载模型
    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if use_tta:
                outputs = tta_predict(model, device, inputs)
            else:
                outputs = model(inputs)
                
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # 计算准确率
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test Accuracy: {acc:.4f}")
    
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # 绘制混淆矩阵
    save_dir = os.path.dirname(model_path)
    cm_path = os.path.join(save_dir, f'confusion_matrix_{"tta" if use_tta else "normal"}.png')
    plot_confusion_matrix(y_true, y_pred, classes, save_path=cm_path)

if __name__ == "__main__":
    # 使用相对路径，方便队友直接运行
    # 假设在 plant-disease-recognition 根目录下运行
    MODEL_PATH = r'results/best_resnet50_mixup.pth' 
    DATA_DIR = r'data/raw'
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please make sure you are in the project root.")
        exit(1)

    # 运行普通评估
    evaluate(MODEL_PATH, DATA_DIR, model_name='resnet50', use_tta=False)
    
    # 运行 TTA 评估 (看看能不能提分)
    print("\nRunning TTA Evaluation...")
    evaluate(MODEL_PATH, DATA_DIR, model_name='resnet50', use_tta=True)
