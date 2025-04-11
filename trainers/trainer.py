import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.config import TRAIN_CONFIG
from utils.visualization import TrainingVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        try:
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.device = device
            self.visualizer = TrainingVisualizer()
            logging.info(f'初始化训练器，使用设备：{device}')
            
            # 记录数据集信息
            logging.info(f'训练集大小：{len(train_loader.dataset)} 批次数：{len(train_loader)}')
            logging.info(f'验证集大小：{len(val_loader.dataset)} 批次数：{len(val_loader)}')
            logging.info(f'测试集大小：{len(test_loader.dataset)} 批次数：{len(test_loader)}')
        except Exception as e:
            logging.error(f'训练器初始化失败：{str(e)}')
            raise
        
        # 初始化训练组件
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=TRAIN_CONFIG['scheduler_factor'],
            patience=TRAIN_CONFIG['scheduler_patience']
        )
        
        # 设置混合精度训练
        self.scaler = torch.amp.GradScaler() if TRAIN_CONFIG['use_mixed_precision'] and device.type == 'cuda' else None
        
        # 训练状态
        self.best_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(self.train_loader, desc='Training'):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                if TRAIN_CONFIG['gradient_clip_val'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAIN_CONFIG['gradient_clip_val'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                if TRAIN_CONFIG['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAIN_CONFIG['gradient_clip_val'])
                self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_loader.dataset)
        
        return epoch_loss, epoch_acc
    
    def test(self):
        self.model.eval()
        test_loss = 0.0
        test_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)
        
        test_loss = test_loss / len(self.test_loader.dataset)
        test_acc = test_corrects.double() / len(self.test_loader.dataset)
        
        return test_loss, test_acc
    
    def train(self, num_epochs):
        try:
            logging.info(f'开始训练，总轮数：{num_epochs}')
            for epoch in range(num_epochs):
                logging.info(f'Epoch {epoch+1}/{num_epochs}')
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f'学习率：{current_lr:.6f}')
                
                # 获取GPU内存使用情况
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**2
                else:
                    gpu_memory = None
                
                # 训练和验证
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()
                
                logging.info(f'训练损失：{train_loss:.4f} 准确率：{train_acc:.4f}')
                logging.info(f'验证损失：{val_loss:.4f} 准确率：{val_acc:.4f}')
                
                # 更新可视化图表
                self.visualizer.update_plots(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    lr=current_lr,
                    gpu_memory=gpu_memory
                )
                
                # 更新学习率
                self.scheduler.step(val_acc)
                
                # 保存最佳模型和早停机制
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    model_save_path = os.path.join(TRAIN_CONFIG['save_dir'], 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_acc': self.best_acc,
                        'scaler': self.scaler.state_dict() if self.scaler else None
                    }, model_save_path)
                    logging.info(f'保存最佳模型，准确率：{self.best_acc:.4f}')
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
                        logging.info(f'触发早停机制，在第 {epoch + 1} 轮停止训练')
                        break
                
            # 记录GPU内存使用情况
            if self.device.type == 'cuda':
                gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**2
                logging.info(f'GPU内存使用：{gpu_memory:.2f} MB')
                
            # 记录当前训练状态
            self._save_checkpoint(epoch, val_acc)
            
            print()
        
            logging.info('-' * 50)
            
        except Exception as e:
            logging.error(f'训练过程出错：{str(e)}')
            raise
        finally:
            # 关闭可视化工具
            self.visualizer.close()
            logging.info('训练结束')
            
    def _save_checkpoint(self, epoch, val_acc):
        """保存检查点"""
        try:
            checkpoint_path = os.path.join(TRAIN_CONFIG['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_acc': val_acc,
                'scaler': self.scaler.state_dict() if self.scaler else None
            }, checkpoint_path)
            logging.info(f'保存检查点：{checkpoint_path}')
        except Exception as e:
            logging.error(f'保存检查点失败：{str(e)}')