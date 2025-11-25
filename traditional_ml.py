import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# 负责人: 队友 B
# 任务: 实现传统机器学习流程 (特征提取 + 分类)

def extract_hog_features(image_path):
    """
    提取 HOG 特征
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 64)) # HOG 需要固定尺寸
    # TODO: 使用 cv2 或 skimage 提取 HOG 特征
    pass

def train_traditional_model(data_dir):
    # 1. 加载数据
    # 2. 提取特征 (X) 和 标签 (y)
    # 3. 训练分类器 (SVM 或 RF)
    clf = SVC(kernel='linear')
    # clf.fit(X_train, y_train)
    
    # 4. 评估
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    pass

if __name__ == "__main__":
    # TODO: 运行训练流程
    pass
