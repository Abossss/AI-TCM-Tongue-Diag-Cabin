import os
import torch
from models import CNN
from models.pretrained_model import load_pretrained_model
from data_processing import get_data_loaders
from trainers import Trainer
from utils.config import TRAIN_CONFIG, DATA_CONFIG, MODEL_CONFIG

def setup_device():
    """设置训练设备"""
    device = torch.device('cuda' if torch.cuda.is_available() and TRAIN_CONFIG['device'] == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(TRAIN_CONFIG['cuda_device_id'])
        print(f'使用GPU: {torch.cuda.get_device_name(device)}')
        print(f'可用GPU数量: {torch.cuda.device_count()}')
        print(f'当前GPU内存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB')
    else:
        print('使用CPU进行训练')
    return device

def main():
    # 创建保存目录
    os.makedirs(TRAIN_CONFIG['save_dir'], exist_ok=True)
    
    # 设置设备
    device = setup_device()
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        DATA_CONFIG['data_dir'],
        TRAIN_CONFIG['batch_size']
    )
    
    # 根据配置选择模型
    if MODEL_CONFIG['use_pretrained']:
        model = load_pretrained_model(MODEL_CONFIG['pretrained_model_path']).to(device)
        print(f"使用预训练模型: {MODEL_CONFIG['model_type']}")
    else:
        model = CNN().to(device)
        print("使用自定义CNN模型")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )
    
    # 训练模型
    trainer.train(num_epochs=TRAIN_CONFIG['num_epochs'])
    
    # 在测试集上评估模型
    print('\n在测试集上评估模型...')
    test_loss, test_acc = trainer.test()
    print(f'测试集结果 - Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

if __name__ == '__main__':
    main()