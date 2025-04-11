# 数据处理流程文档

## 1. 数据集结构

### 1.1 目录组织
```
data/
├── 淡白舌灰黑苔/
│   ├── 拍摄不规范/
│   └── ...
├── 淡白舌白苔/
│   ├── 淡白舌白苔厚苔腻苔/
│   ├── 淡白舌白苔厚苔非腻苔/
│   ├── 淡白舌白苔薄苔腻苔/
│   └── 淡白舌白苔薄苔非腻苔/
├── 淡红舌灰黑苔/
│   ├── 淡红舌灰黑苔厚苔腻苔/
│   ├── 淡红舌灰黑苔厚苔非腻苔/
│   └── ...
└── ...
```

### 1.2 类别体系
- **舌质分类**
  - 淡白舌
  - 淡红舌
  - 红舌
  - 绛舌
  - 青紫舌

- **舌苔分类**
  - 白苔
  - 黄苔
  - 灰黑苔

- **舌苔特征**
  - 厚薄
    - 厚苔
    - 薄苔
  - 腻非腻
    - 腻苔
    - 非腻苔

## 2. 数据预处理

### 2.1 图像标准化
```python
transforms.Compose([
    transforms.Resize((224, 224)),  # 统一尺寸
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 2.2 标签处理
- 多级标签编码
  - 舌质标签
  - 舌苔标签
  - 特征标签

### 2.3 数据清洗
- 去除低质量图像
- 处理错误标注
- 剔除重复样本

## 3. 数据增强

### 3.1 空间变换
```python
transforms.RandomApply([
    transforms.RandomRotation(15),    # 随机旋转
    transforms.RandomHorizontalFlip(), # 水平翻转
    transforms.RandomVerticalFlip(),   # 垂直翻转
    transforms.RandomCrop(224, padding=32) # 随机裁剪
], p=0.7)
```

### 3.2 颜色变换
```python
transforms.ColorJitter(
    brightness=0.2,  # 亮度调整
    contrast=0.2,    # 对比度调整
    saturation=0.2,  # 饱和度调整
    hue=0.1          # 色调调整
)
```

### 3.3 噪声注入
- 高斯噪声
- 椒盐噪声
- 模糊效果

### 3.4 遮挡增强
```python
transforms.RandomErasing(
    p=0.5,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3)
)
```

## 4. 数据加载

### 4.1 数据集类设计
```python
class TongueDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels
```

### 4.2 数据批处理
```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 4.3 采样策略
- 类别平衡采样
- 难例挖掘
- 动态采样

## 5. 数据集划分

### 5.1 划分策略
- 训练集：70%
- 验证集：20%
- 测试集：10%

### 5.2 划分方法
```python
def split_dataset(root_dir, train_ratio=0.7, val_ratio=0.2):
    # 确保比例合理
    assert train_ratio + val_ratio < 1.0
    
    # 获取所有样本
    all_samples = []
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                if img.endswith(('.jpg', '.png')):
                    all_samples.append((os.path.join(class_path, img), class_dir))
    
    # 随机打乱
    random.shuffle(all_samples)
    
    # 计算划分点
    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分数据集
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train+n_val]
    test_samples = all_samples[n_train+n_val:]
    
    return train_samples, val_samples, test_samples
```

## 6. 数据验证

### 6.1 数据完整性检查
- 文件完整性
- 标签一致性
- 图像可读性

### 6.2 类别分布分析
- 类别统计
- 分布可视化
- 不平衡检测

### 6.3 图像质量评估
- 分辨率检查
- 清晰度评估
- 光照条件分析

## 7. 性能优化

### 7.1 数据加载优化
- 多进程加载
- 内存映射
- 预取机制

### 7.2 缓存策略
- 数据缓存
- 特征缓存
- 批处理缓存

### 7.3 内存管理
- 渐进式加载
- 动态释放
- 内存监控

## 8. 数据安全

### 8.1 数据备份
- 定期备份
- 增量备份
- 版本控制

### 8.2 隐私保护
- 数据脱敏
- 访问控制
- 加密存储

### 8.3 数据审计
- 操作日志
- 使用统计
- 异常检测