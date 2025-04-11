# 项目结构说明

```
CNN/
├── data/                   # 数据集目录
├── models/                 # 模型相关代码
│   ├── __init__.py
│   ├── cnn.py             # CNN模型定义
│   └── attention.py       # 注意力机制模块
├── data_processing/        # 数据处理相关代码
│   ├── __init__.py
│   ├── dataset.py         # 数据集加载和预处理
│   └── augmentation.py    # 数据增强配置
├── trainers/              # 训练相关代码
│   ├── __init__.py
│   └── trainer.py         # 训练器实现
├── utils/                 # 工具类代码
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   └── visualization.py   # 可视化工具
├── main.py                # 主程序入口
└── requirements.txt       # 项目依赖
```

## 目录说明

- `data/`: 存放数据集文件
- `models/`: 存放模型相关的代码，包括CNN模型定义和注意力机制模块
- `data_processing/`: 数据处理相关代码，包括数据集加载和数据增强
- `trainers/`: 训练相关代码，包括训练循环和评估逻辑
- `utils/`: 工具类代码，包括配置文件和可视化工具
- `main.py`: 主程序入口，用于启动训练或预测
- `requirements.txt`: 项目依赖文件