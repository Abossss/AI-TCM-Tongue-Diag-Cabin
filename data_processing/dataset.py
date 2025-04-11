import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.config import DATA_CONFIG
from .augmentation import get_train_transform, get_eval_transform

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'数据目录不存在：{data_dir}')
            
        self.data_dir = data_dir
        self.transform = transform
        
        # 获取并验证类别
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not self.classes:
            raise ValueError(f'数据目录 {data_dir} 中没有有效的类别子目录')
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        self.class_counts = {cls: 0 for cls in self.classes}
        
        def scan_directory(dir_path, class_name):
            """递归扫描目录下的所有图片文件"""
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    if item != '拍摄不规范':  # 跳过特定的目录
                        scan_directory(item_path, class_name)
                elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # 验证图片是否可以打开
                        with Image.open(item_path) as img:
                            img.verify()
                        self.images.append(item_path)
                        self.labels.append(self.class_to_idx[class_name])
                        self.class_counts[class_name] += 1
                    except Exception as e:
                        print(f'警告：图片 {item_path} 无效或已损坏：{str(e)}')

        # 加载并验证数据
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            scan_directory(class_dir, class_name)
            
            if self.class_counts[class_name] == 0:
                print(f'警告：类别 {class_name} 中没有有效的图片文件')

                    
        if not self.images:
            raise ValueError('没有找到有效的图片文件')
            
        # 打印数据集统计信息
        print('\n数据集统计信息：')
        print(f'总类别数：{len(self.classes)}')
        print(f'总图片数：{len(self.images)}')
        for cls, count in self.class_counts.items():
            print(f'类别 {cls}：{count} 张图片')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=None, train_transform=None, eval_transform=None):
    """创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小，如果为None则使用配置文件中的值
        train_transform: 训练数据的转换，如果为None则使用默认转换
        eval_transform: 评估数据的转换，如果为None则使用默认转换
    
    Returns:
        train_loader, val_loader, test_loader: 数据加载器元组
    """
    if batch_size is None:
        from utils.config import TRAIN_CONFIG
        batch_size = TRAIN_CONFIG['batch_size']
        
    if train_transform is None:
        train_transform = get_train_transform()
    if eval_transform is None:
        eval_transform = get_eval_transform()
    
    try:
        # 创建训练、验证和测试数据集实例
        train_dataset = ImageDataset(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = ImageDataset(os.path.join(data_dir, 'val'), transform=eval_transform)
        test_dataset = ImageDataset(os.path.join(data_dir, 'test'), transform=eval_transform)
    
    # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=DATA_CONFIG['num_workers'],
            pin_memory=DATA_CONFIG['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DATA_CONFIG['num_workers'],
            pin_memory=DATA_CONFIG['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DATA_CONFIG['num_workers'],
            pin_memory=DATA_CONFIG['pin_memory']
        )
        
        print(f'\n数据加载器创建完成：')
        print(f'训练集大小：{len(train_dataset)} 批次数：{len(train_loader)}')
        print(f'验证集大小：{len(val_dataset)} 批次数：{len(val_loader)}')
        print(f'测试集大小：{len(test_dataset)} 批次数：{len(test_loader)}')
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        raise Exception(f'创建数据加载器失败：{str(e)}')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DATA_CONFIG['num_workers'],
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATA_CONFIG['num_workers'],
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATA_CONFIG['num_workers'],
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    return train_loader, val_loader, test_loader