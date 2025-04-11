# 模型架构设计文档

## 1. CNN基础模型

### 1.1 整体架构
```python
Input Image (3x224x224)
    ↓
Conv Block 1 (64 channels)
    Conv2d(3→64, kernel=3, stride=1)
    BatchNorm2d
    ReLU
    MaxPool2d(2×2)
    ↓
Conv Block 2 (128 channels)
    Conv2d(64→128, kernel=3, stride=1)
    BatchNorm2d
    ReLU
    MaxPool2d(2×2)
    ↓
Conv Block 3 (256 channels)
    Conv2d(128→256, kernel=3, stride=1)
    BatchNorm2d
    ReLU
    MaxPool2d(2×2)
    ↓
Attention Module
    ↓
Global Average Pooling
    ↓
Fully Connected Layers
    FC1: 256 → 128
    ReLU + Dropout(0.5)
    FC2: 128 → 64
    ReLU + Dropout(0.3)
    FC3: 64 → num_classes
```

### 1.2 卷积块设计
每个卷积块的详细参数：
- **Conv Block 1**
  - 输入：3通道
  - 输出：64通道
  - 感受野：3×3
  - 步长：1
  - 填充：1

- **Conv Block 2**
  - 输入：64通道
  - 输出：128通道
  - 感受野：3×3
  - 步长：1
  - 填充：1

- **Conv Block 3**
  - 输入：128通道
  - 输出：256通道
  - 感受野：3×3
  - 步长：1
  - 填充：1

### 1.3 特征提取流程
1. **第一阶段特征提取**
   - 提取低级特征（边缘、纹理）
   - 特征图尺寸：112×112

2. **第二阶段特征提取**
   - 提取中级特征（形状、局部结构）
   - 特征图尺寸：56×56

3. **第三阶段特征提取**
   - 提取高级特征（语义信息）
   - 特征图尺寸：28×28

## 2. 注意力机制

### 2.1 自注意力模块
```python
Input Feature Maps
    ↓
Query/Key/Value Transform
    ↓
Attention Weights Calculation
    ↓
Feature Refinement
    ↓
Output Feature Maps
```

### 2.2 通道注意力
- **Squeeze操作**：全局平均池化
- **Excitation操作**：两层全连接网络
  - FC1: C → C/r (r=16)
  - ReLU
  - FC2: C/r → C
  - Sigmoid

### 2.3 空间注意力
- 通道维度压缩：最大池化和平均池化
- 特征融合：7×7卷积
- 注意力权重：Sigmoid激活

## 3. 分类头设计

### 3.1 全连接层
- **第一层**
  - 输入：256
  - 输出：128
  - Dropout率：0.5

- **第二层**
  - 输入：128
  - 输出：64
  - Dropout率：0.3

- **输出层**
  - 输入：64
  - 输出：类别数

### 3.2 激活函数选择
- 隐藏层：ReLU
  - 优点：解决梯度消失
  - 计算效率高

- 输出层：Softmax
  - 多类别分类
  - 概率输出

## 4. 模型优化策略

### 4.1 正则化方法
- Dropout
- BatchNormalization
- L2正则化（权重衰减）

### 4.2 初始化策略
- 卷积层：He初始化
- 全连接层：Xavier初始化

### 4.3 训练技巧
- 学习率调度
- 梯度裁剪
- 早停策略

## 5. 模型评估

### 5.1 计算复杂度
- 模型参数量：约2M
- FLOPs：约900M
- 推理时间：<100ms (GPU)

### 5.2 内存占用
- 训练时显存：~4GB
- 推理时显存：~2GB
- 模型大小：~8MB

### 5.3 性能指标
- Top-1准确率：92%
- Top-5准确率：98%
- 混淆矩阵分析

## 6. 模型部署

### 6.1 模型导出
- ONNX格式
- TorchScript
- TensorRT优化

### 6.2 量化策略
- 动态量化
- 静态量化
- 量化感知训练

### 6.3 加速方法
- 模型剪枝
- 知识蒸馏
- 架构优化