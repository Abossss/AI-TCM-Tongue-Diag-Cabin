import unittest
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CNN
from dataset import get_data_loaders

class TestCNN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_classes = 10
        self.input_shape = (3, 224, 224)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN(self.num_classes).to(self.device)
    
    def test_model_output_shape(self):
        # 测试模型输出维度是否正确
        x = torch.randn(self.batch_size, *self.input_shape).to(self.device)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_model_forward_pass(self):
        # 测试模型前向传播是否正常工作
        x = torch.randn(self.batch_size, *self.input_shape).to(self.device)
        try:
            output = self.model(x)
            # 检查输出是否包含NaN
            self.assertFalse(torch.isnan(output).any())
        except Exception as e:
            self.fail(f"模型前向传播失败: {str(e)}")
    
    def test_feature_map_shapes(self):
        # 测试每个卷积层的特征图尺寸
        x = torch.randn(self.batch_size, *self.input_shape).to(self.device)
        
        # 第一个卷积层后的尺寸 (224/2 = 112)
        x1 = self.model.conv1(x)
        self.assertEqual(x1.shape, (self.batch_size, 64, 112, 112))
        
        # 第二个卷积层后的尺寸 (112/2 = 56)
        x2 = self.model.conv2(x1)
        self.assertEqual(x2.shape, (self.batch_size, 128, 56, 56))
        
        # 第三个卷积层后的尺寸 (56/2 = 28)
        x3 = self.model.conv3(x2)
        self.assertEqual(x3.shape, (self.batch_size, 256, 28, 28))

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # 创建临时测试数据目录
        self.test_data_dir = 'test_data'
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.batch_size = 4
    
    def tearDown(self):
        # 清理测试数据目录
        if os.path.exists(self.test_data_dir):
            import shutil
            shutil.rmtree(self.test_data_dir)
    
    def test_data_loader_split(self):
        # 测试数据集划分是否正确
        train_loader, val_loader = get_data_loaders(self.test_data_dir, self.batch_size)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

if __name__ == '__main__':
    unittest.main()