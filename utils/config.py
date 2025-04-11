# 数据集配置
DATA_CONFIG = {
    'data_dir': 'pre_data',  # 使用预处理后的数据目录
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'random_seed': 42,
    'num_workers': 2,
    'pin_memory': True
}

# 模型配置
MODEL_CONFIG = {
    'num_classes': 60,  # 更新为新数据集的类别数量
    'input_channels': 3,
    'dropout_rate': 0.5,  # 增加dropout率以减少过拟合
    'use_pretrained': True,  # 是否使用预训练模型
    'pretrained_model_path': None,  # 预训练模型路径，None表示使用ImageNet预训练权重
    'model_type': 'resnet152'  # 使用的模型类型
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,  # 减小批次大小以提高稳定性
    'num_epochs': 150,  # 增加训练轮数
    'learning_rate': 0.0001,  # 进一步降低初始学习率
    'weight_decay': 0.001,   # 增加权重衰减以加强正则化
    'save_dir': 'checkpoints',
    'early_stopping_patience': 15,  # 增加早停耐心值
    'scheduler_patience': 8,        # 增加调度器耐心值
    'scheduler_factor': 0.1,        # 减小学习率衰减因子以实现更平缓的学习率下降
    'device': 'cuda',               # 使用的设备，可选 'cuda' 或 'cpu'
    'cuda_device_id': 0,           # 使用的CUDA设备ID
    'use_mixed_precision': True,    # 是否使用混合精度训练
    'gradient_clip_val': 1.5        # 梯度裁剪值
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'resize_size': 256,
    'crop_size': 224,
    'rotation_degrees': 20,
    'brightness': 0.3,
    'contrast': 0.3,
    'saturation': 0.3,
    'hue': 0.15
}

# 标准化参数
NORMALIZATION_CONFIG = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}