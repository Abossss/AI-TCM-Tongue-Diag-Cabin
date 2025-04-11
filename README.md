<div align="center">

# <span style="color: #2c7be5;">PyTorch CNN Image Classification Project 🚀🧠</span>

[简体中文](README_CN.md) / [繁体中文](README_TC.md) / English / [Deutsch](README_DE.md) / [日本語](README_JP.md)

</div>  

This is a CNN image classification project implemented using the PyTorch framework, providing a complete training process and data processing functions. The project integrates an attention mechanism, supports multiple data augmentation methods, and offers a complete training and evaluation process.

## <span style="color: #228be6;">Project Structure 📁🗂️</span>

```
├── models/             # Model-related definitions
│   ├── cnn.py         # Basic CNN model
│   └── attention.py   # Attention mechanism module
├── data_processing/    # Data processing-related
│   └── dataset.py     # Dataset loading and preprocessing
├── trainers/          # Training-related
│   └── trainer.py     # Trainer implementation
├── utils/             # Utility functions
│   ├── config.py      # Configuration management
│   └── visualization.py # Visualization tools
├── tests/             # Test code
│   └── test_model.py  # Model testing
├── static/            # Static resources
├── templates/         # Web templates
├── predict.py         # Prediction script
├── main.py           # Main program entry
└── requirements.txt   # Project dependencies
```

## <span style="color: #228be6;">Main Features ✨🌟</span>

<span style="color: #38d9a9;">- Integrated attention mechanism to improve model performance</span>
<span style="color: #38d9a9;">- Support for multiple data augmentation methods</span>
<span style="color: #38d9a9;">- Provide a Web interface for online prediction</span>
<span style="color: #38d9a9;">- Support for model training process visualization</span>
<span style="color: #38d9a9;">- Complete test cases</span>

## <span style="color: #228be6;">Environment Configuration ⚙️🛠️</span>

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. CUDA support (recommended):
   - Ensure that the NVIDIA GPU driver is installed
   - PyTorch will automatically detect and use available GPUs

## <span style="color: #228be6;">Data Preparation 📊📂</span>

1. Organize the dataset in the `data` directory:
   - Create a subfolder for each category
   - Place the corresponding category's images into the respective folder

Example structure:
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

## <span style="color: #228be6;">Training the Model 🧠💪</span>

1. Configure training parameters
   Edit the `utils/config.py` file:
   ```python
   # Set training parameters
   num_classes = 15  # Number of classification categories
   batch_size = 32   # Batch size
   num_epochs = 100  # Number of training epochs
   learning_rate = 0.001  # Learning rate
   ```

2. Start training:
   ```bash
   python main.py --mode train
   ```

3. Training process visualization:
   - Loss curves and accuracy curves are updated in real-time
   - Model checkpoints are automatically saved in the `checkpoints` directory

## <span style="color: #228be6;">Using the Model for Prediction 🎯✅</span>

### Web Interface Prediction

1. Start the Web service:
   ```bash
   python app.py
   ```

2. Visit `http://localhost:5000` for online prediction

### Command Line Prediction

```python
from predict import ImagePredictor

# Initialize the predictor
predictor = ImagePredictor('checkpoints/best_model.pth')

# Single image prediction
result = predictor.predict_single('data/ak47/001_0001.jpg')
print(f'Predicted class: {result["class