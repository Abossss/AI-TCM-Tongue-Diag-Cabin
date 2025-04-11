import os
import shutil
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_CONFIG
from data_processing.dataset import ImageDataset
from data_processing.augmentation import get_train_transform

def split_and_save_dataset(data_dir, output_dir='pre_data'):
    """将数据集分割并保存到指定目录
    
    Args:
        data_dir: 原始数据目录路径
        output_dir: 输出目录路径，默认为'pre_data'
    """
    try:
        # 创建数据集实例
        dataset = ImageDataset(data_dir, transform=None)  # 不需要转换，因为我们只需要文件路径
        
        # 设置随机种子以确保可重复性
        torch.manual_seed(DATA_CONFIG['random_seed'])
        
        # 计算数据集分割大小
        total_size = len(dataset)
        train_size = int(total_size * DATA_CONFIG['train_ratio'])
        val_size = int(total_size * DATA_CONFIG['val_ratio'])
        test_size = total_size - train_size - val_size
        
        # 分割数据集
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')
        
        # 创建分割后的目录结构
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(split_dir, exist_ok=True)
            # 为每个类别创建子目录
            for class_name in dataset.classes:
                os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        def copy_dataset_files(split_dataset, target_dir):
            """复制数据集文件到目标目录"""
            for idx in split_dataset.indices:
                src_path = dataset.images[idx]
                label = dataset.labels[idx]
                class_name = dataset.classes[label]
                
                # 获取文件名
                filename = os.path.basename(src_path)
                
                # 构建目标路径
                dst_path = os.path.join(target_dir, class_name, filename)
                
                # 复制文件
                shutil.copy2(src_path, dst_path)
        
        # 复制文件到对应目录
        print('正在复制训练集文件...')
        copy_dataset_files(train_dataset, train_dir)
        
        print('正在复制验证集文件...')
        copy_dataset_files(val_dataset, val_dir)
        
        print('正在复制测试集文件...')
        copy_dataset_files(test_dataset, test_dir)
        
        # 打印统计信息
        print('\n数据集分割完成：')
        print(f'训练集大小：{len(train_dataset)} 图片')
        print(f'验证集大小：{len(val_dataset)} 图片')
        print(f'测试集大小：{len(test_dataset)} 图片')
        print(f'\n文件已保存到：{os.path.abspath(output_dir)}')
        
    except Exception as e:
        raise Exception(f'数据集分割失败：{str(e)}')

if __name__ == '__main__':
    # 使用相对路径，与config.py中的配置保持一致
    split_and_save_dataset(DATA_CONFIG['data_dir'])