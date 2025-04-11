import torch
import torch.nn as nn
from utils.config import MODEL_CONFIG
from .attention import SelfAttention

class CNN(nn.Module):
    def __init__(self, num_classes=MODEL_CONFIG['num_classes']):
        super(CNN, self).__init__()
        
        # 卷积层块1
        self.conv1 = nn.Sequential(
            nn.Conv2d(MODEL_CONFIG['input_channels'], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(MODEL_CONFIG['dropout_rate']/2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 卷积层块2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(MODEL_CONFIG['dropout_rate']/2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention1 = SelfAttention(128)
        
        # 卷积层块3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(MODEL_CONFIG['dropout_rate']/2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention2 = SelfAttention(256)
        
        # 卷积层块4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(MODEL_CONFIG['dropout_rate']/2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention3 = SelfAttention(512)
        
        # 计算全连接层的输入维度
        self.fc_input_dim = 512 * 14 * 14  # 224/16 = 14
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        # 应用卷积层和注意力机制
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention1(x)
        x = self.conv3(x)
        x = self.attention2(x)
        x = self.conv4(x)
        x = self.attention3(x)
        
        # 展平特征图
        x = x.view(x.size(0), -1)
        
        # 应用全连接层
        x = self.fc(x)
        
        return x