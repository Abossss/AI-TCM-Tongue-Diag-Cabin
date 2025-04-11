import torch
import torch.nn as nn
from torchvision import models
from utils.config import MODEL_CONFIG

class PretrainedCNN(nn.Module):
    def __init__(self, num_classes=MODEL_CONFIG['num_classes'], model_name='resnet152', pretrained=True):
        super(PretrainedCNN, self).__init__()
        
        # 加载预训练模型
        if model_name == 'resnet152':
            self.base_model = models.resnet152(weights='IMAGENET1K_V2' if pretrained else None)
            feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # 移除最后的全连接层
        
        # 添加自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        # 通过基础网络提取特征
        features = self.base_model(x)
        # 应用分类头
        output = self.classifier(features)
        return output

def load_pretrained_model(model_path=None):
    """加载预训练模型
    
    Args:
        model_path: 预训练模型权重文件路径，如果为None则使用ImageNet预训练权重
        
    Returns:
        model: 加载了预训练权重的模型
    """
    model = PretrainedCNN()
    
    if model_path:
        # 加载自定义预训练权重
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Successfully loaded model from {model_path}')
    else:
        print('Using ImageNet pretrained weights')
    
    return model

def get_data_transforms():
    """获取数据预处理转换
    
    Returns:
        transforms_dict: 包含训练和验证阶段的数据转换
    """
    from torchvision import transforms
    
    # ImageNet预训练模型的标准化参数
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transforms_dict = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    }
    
    return transforms_dict