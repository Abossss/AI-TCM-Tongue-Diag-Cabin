import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from models.pretrained_model import PretrainedCNN, get_data_transforms
from utils.config import MODEL_CONFIG
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predict.log'),
        logging.StreamHandler()
    ]
)

class ImagePredictor:
    def __init__(self, model_path, num_classes=MODEL_CONFIG['num_classes']):
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logging.error(f'模型文件不存在：{model_path}')
            raise FileNotFoundError(f'模型文件不存在：{model_path}')
            
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'使用设备: {self.device}')
        
        try:
            self.model = PretrainedCNN(num_classes=num_classes, model_name='resnet152', pretrained=False).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 尝试直接加载状态字典
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            logging.info('模型加载成功')
        except Exception as e:
            logging.error(f'模型加载失败：{str(e)}')
            raise Exception(f'模型加载失败：{str(e)}')
        
        # 获取预训练模型的预处理转换
        transforms_dict = get_data_transforms()
        self.transform = transforms_dict['val']  # 使用验证集的转换
        
        # 从pre_data目录获取类别名称
        data_dir = 'pre_data/train'  # 使用训练集目录获取类别信息
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'数据目录不存在：{data_dir}')
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not self.classes:
            raise ValueError('没有找到有效的类别目录')
        logging.info(f'加载了 {len(self.classes)} 个类别')
        for cls in self.classes:
            logging.debug(f'类别: {cls}')
    
    def preprocess_image(self, image_path):
        """预处理单张图片"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'图片文件不存在：{image_path}')
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image.unsqueeze(0)  # 添加batch维度
        except Exception as e:
            raise Exception(f'图片预处理失败：{str(e)}')
    
    def predict_single(self, image_path):
        """预测单张图片"""
        # 预处理图片
        image = self.preprocess_image(image_path)
        image = image.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            probability = torch.nn.functional.softmax(outputs, dim=1)
            
        # 获取预测结果
        pred_class = self.classes[predicted.item()]
        pred_prob = probability[0][predicted.item()].item()
        
        return {
            'class': pred_class,
            'probability': f'{pred_prob:.2%}'
        }
    
    def predict_batch(self, image_dir):
        """预测文件夹中的所有图片"""
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f'图片目录不存在：{image_dir}')
            
        results = []
        errors = []
        
        # 遍历文件夹中的所有图片
        for image_name in os.listdir(image_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, image_name)
                try:
                    prediction = self.predict_single(image_path)
                    results.append({
                        'image': image_name,
                        'prediction': prediction
                    })
                except Exception as e:
                    errors.append({
                        'image': image_name,
                        'error': str(e)
                    })
        
        return results

def main():
    try:
        # 使用训练好的模型
        model_path = 'checkpoints/best_model.pth'
        
        # 创建预测器实例
        predictor = ImagePredictor(model_path)
        
        # 从pre_data/test目录中选择一个类别进行预测
        test_categories = [d for d in os.listdir('pre_data/test') if os.path.isdir(os.path.join('pre_data/test', d))]
        if not test_categories:
            logging.error('测试数据目录为空')
            return
        
        # 选择第一个类别进行演示
        test_category = test_categories[0]
        test_dir = os.path.join('pre_data/test', test_category)
        
        logging.info(f'\n选择测试类别: {test_category}')
        
        # 批量预测该类别下的所有图片
        batch_results = predictor.predict_batch(test_dir)
        
        if not batch_results:
            logging.info(f'目录 {test_dir} 中没有找到有效的图片文件')
            return
        
        logging.info('\n预测结果:')
        for result in batch_results:
            logging.info(f'图片: {result["image"]}')
            logging.info(f'预测类别: {result["prediction"]["class"]}')
            logging.info(f'置信度: {result["prediction"]["probability"]}')
            
    except Exception as e:
        logging.error(f'预测过程出错：{str(e)}')
        raise

if __name__ == '__main__':
    main()