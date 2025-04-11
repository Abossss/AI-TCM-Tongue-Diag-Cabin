# pre_data 数据集说明

## 数据集来源
本数据集中的数据是通过分割函数从原始数据集中分割而来，原始数据集的详细信息请参考项目根目录下的数据相关文档。

## 分割方式
数据分割使用了 `data_processing/split_dataset.py` 脚本进行操作。具体的分割逻辑和参数可以查看该脚本文件。

## 文件结构
```
pre_data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
├── val/
│   ├── class1/
│   │   ├── image5.jpg
│   │   └── image6.jpg
│   └── class2/
│       ├── image7.jpg
│       └── image8.jpg
└── test/
    ├── class1/
    │   ├── image9.jpg
    │   └── image10.jpg
    └── class2/
        ├── image11.jpg
        └── image12.jpg
```

## 使用方法
### 训练模型
在训练模型时，可以使用 `train/` 目录下的数据作为训练集。在 `utils/config.py` 中配置好数据集路径后，运行 `python main.py --mode train` 即可开始训练。

### 验证模型
在验证模型时，可以使用 `val/` 目录下的数据作为验证集。模型在训练过程中会自动使用该验证集进行验证。

### 测试模型
在测试模型时，可以使用 `test/` 目录下的数据作为测试集。运行 `python predict.py` 并指定测试集路径即可进行测试。