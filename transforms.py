from torchvision import transforms

# 负责人: 队友 B
# 说明: 队友 A 为了跑通训练流程，提供了一个基础版本。
# 任务: 队友 B 请在此基础上优化数据增强策略 (如添加 ColorJitter, Rotation 等)

# 这个文件会被 src/dataset.py 引用

def get_transforms(split='train'):
    """
    Args:
        split (str): 'train' (需要增强) 或 'val'/'test' (仅需 Resize/Normalize)
    """
    # ImageNet 标准化参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 随机大小裁剪
            transforms.RandomHorizontalFlip(p=0.5),              # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.5),                # 随机垂直翻转
            transforms.RandomRotation(15),                       # 随机旋转 +/- 15度
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # 颜色抖动
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
