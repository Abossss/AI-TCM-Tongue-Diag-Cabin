<div align="center">

# <span style="color: #2c7be5;">PyTorch CNN 图像分类项目 🚀🧠</span>

简体中文 / [繁体中文](README_TC.md) / [English](README.md) / [Deutsch](README_DE.md) / [日本語](README_JP.md)

</div>  

这是一个使用PyTorch框架实现的CNN图像分类项目，提供了完整的训练流程和数据处理功能。项目集成了注意力机制，支持多种数据增强方法，并提供了完整的训练和评估流程。

## <span style="color: #228be6;">项目结构 📁🗂️</span>

```
├── models/             # 模型相关定义
│   ├── cnn.py         # CNN基础模型
│   └── attention.py   # 注意力机制模块
├── data_processing/    # 数据处理相关
│   └── dataset.py     # 数据集加载和预处理
├── trainers/          # 训练相关
│   └── trainer.py     # 训练器实现
├── utils/             # 工具函数
│   ├── config.py      # 配置管理
│   └── visualization.py # 可视化工具
├── tests/             # 测试代码
│   └── test_model.py  # 模型测试
├── static/            # 静态资源
├── templates/         # 网页模板
├── predict.py         # 预测脚本
├── main.py           # 主程序入口
└── requirements.txt   # 项目依赖
```

## <span style="color: #228be6;">主要特性 ✨🌟</span>

<span style="color: #38d9a9;">- 集成注意力机制，提升模型性能</span>
<span style="color: #38d9a9;">- 支持多种数据增强方法</span>
<span style="color: #38d9a9;">- 提供Web界面进行在线预测</span>
<span style="color: #38d9a9;">- 支持模型训练过程可视化</span>
<span style="color: #38d9a9;">- 完整的测试用例</span>

## <span style="color: #228be6;">环境配置 ⚙️🛠️</span>

1. 创建并激活虚拟环境（推荐）：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. CUDA支持（推荐）：
   - 确保已安装NVIDIA GPU驱动
   - PyTorch会自动检测并使用可用的GPU

## <span style="color: #228be6;">数据准备 📊📂</span>

1. 在`data`目录下组织数据集：
   - 每个类别创建一个子文件夹
   - 将对应类别的图片放入相应文件夹

示例结构：
```
data/
  ├── 淡白舌灰黑苔/
  │   ├── example1.jpg
  │   └── example2.jpg
  ├── 淡白舌白苔/
  │   ├── example1.jpg
  │   └── example2.jpg
  └── ...
```

## <span style="color: #228be6;">训练模型 🧠💪</span>

1. 配置训练参数
   编辑`utils/config.py`文件：
   ```python
   # 设置训练参数
   num_classes = 15  # 分类类别数
   batch_size = 32   # 批次大小
   num_epochs = 100  # 训练轮数
   learning_rate = 0.001  # 学习率
   ```

2. 启动训练：
   ```bash
   python main.py --mode train
   ```

3. 训练过程可视化：
   - 损失曲线和准确率曲线实时更新
   - 模型检查点自动保存在`checkpoints`目录

## <span style="color: #228be6;">使用模型预测 🎯✅</span>

### Web界面预测

1. 启动Web服务：
   ```bash
   python app.py
   ```

2. 访问`http://localhost:5000`进行在线预测

### 命令行预测

```python
from predict import ImagePredictor

# 初始化预测器
predictor = ImagePredictor('checkpoints/best_model.pth')

# 单张图片预测
result = predictor.predict_single('data/ak47/001_0001.jpg')
print(f'预测类别: {result["class"]}')
print(f'置信度: {result["probability"]}')

# 批量预测
results = predictor.predict_batch('data/ak47')
for result in results:
    print(f'图片: {result["image"]}')
    print(f'预测结果: {result["prediction"]}')
```

## <span style="color: #228be6;">模型架构 🏗️</span>

- 基础CNN架构：3个卷积层块（卷积+批归一化+ReLU+池化）
- 注意力机制：自注意力模块，增强特征提取能力
- 全连接层：3层用于特征降维和分类
- Dropout层：防止过拟合
- 损失函数：交叉熵损失
- 优化器：Adam

## <span style="color: #228be6;">注意事项 ⚠️</span>

- 支持的图片格式：jpg、jpeg、png
- 推荐使用GPU进行训练
- 可通过修改配置文件调整模型结构和超参数
- 定期备份训练好的模型文件
- 预测时确保模型文件路径正确