<div align="center">

# <span style="color: #2c7be5;">PyTorch CNN 圖像分類項目 🚀🧠</span>

[简体中文](README_CN.md) / 繁体中文 / [English](README.md) / vs / [日本語](README_JP.md)

</div>

這是一個使用PyTorch框架實現的CNN圖像分類項目，提供了完整的訓練流程和數據處理功能。項目集成了注意力機制，支援多種數據增強方法，並提供了完整的訓練和評估流程。

## <span style="color: #228be6;">項目結構 📁🗂️</span>

```
├── models/             # 模型相關定義
│   ├── cnn.py         # CNN基礎模型
│   └── attention.py   # 注意力機制模塊
├── data_processing/    # 數據處理相關
│   └── dataset.py     # 數據集加載和預處理
├── trainers/          # 訓練相關
│   └── trainer.py     # 訓練器實現
├── utils/             # 工具函數
│   ├── config.py      # 配置管理
│   └── visualization.py # 可視化工具
├── tests/             # 測試代碼
│   └── test_model.py  # 模型測試
├── static/            # 靜態資源
├── templates/         # 網頁模板
├── predict.py         # 預測腳本
├── main.py           # 主程序入口
└── requirements.txt   # 項目依賴
```

## <span style="color: #228be6;">主要特性 ✨🌟</span>

<span style="color: #38d9a9;">- 集成注意力機制，提升模型性能</span>
<span style="color: #38d9a9;">- 支援多種數據增強方法</span>
<span style="color: #38d9a9;">- 提供Web界面進行在線預測</span>
<span style="color: #38d9a9;">- 支援模型訓練過程可視化</span>
<span style="color: #38d9a9;">- 完整的測試用例</span>

## <span style="color: #228be6;">環境配置 ⚙️🛠️</span>

1. 創建並激活虛擬環境（推薦）：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

3. CUDA支援（推薦）：
   - 確保已安裝NVIDIA GPU驅動
   - PyTorch會自動檢測並使用可用的GPU

## <span style="color: #228be6;">數據準備 📊📂</span>

1. 在`data`目錄下組織數據集：
   - 每個類別創建一個子文件夾
   - 將對應類別的圖片放入相應文件夾

示例結構：
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

## <span style="color: #228be6;">訓練模型 🧠💪</span>

1. 配置訓練參數
   編輯`utils/config.py`文件：
   ```python
   # 設置訓練參數
   num_classes = 15  # 分類類別數
   batch_size = 32   # 批次大小
   num_epochs = 100  # 訓練輪數
   learning_rate = 0.001  # 學習率
   ```

2. 啟動訓練：
   ```bash
   python main.py --mode train
   ```

3. 訓練過程可視化：
   - 損失曲線和準確率曲線實時更新
   - 模型檢查點自動保存在`checkpoints`目錄

## <span style="color: #228be6;">使用模型預測 🎯✅</span>

### Web界面預測

1. 啟動Web服務：
   ```bash
   python app.py
   ```

2. 訪問`http://localhost:5000`進行在線預測

### 命令行預測

```python
from predict import ImagePredictor

# 初始化預測器
predictor = ImagePredictor('checkpoints/best_model.pth')

# 單張圖片預測
result = predictor.predict_single('data/ak47/001_0001.jpg')
print(f'預測類別: {result["class"]}')
print(f'置信度: {result["probability"]}')

# 批量預測
results = predictor.predict_batch('data/ak47')
for result in results:
    print(f'圖片: {result["image"]}')
    print(f'預測結果: {result["prediction"]}')
```

## <span style="color: #228be6;">模型架構 🏗️</span>

- 基礎CNN架構：3個卷積層塊（卷積+批歸一化+ReLU+池化）
- 注意力機制：自注意力模塊，增強特徵提取能力
- 全連接層：3層用於特徵降維和分類
- Dropout層：防止過擬合
- 損失函數：交叉熵損失
- 優化器：Adam

## <span style="color: #228be6;">注意事項 ⚠️</span>

- 支援的圖片格式：jpg、jpeg、png
- 推薦使用GPU進行訓練
- 可通過修改配置文件調整模型結構和超參數
- 定期備份訓練好的模型文件
- 預測時確保模型文件路徑正確