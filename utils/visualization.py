import os
import torch
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self, log_dir: str = 'runs/training'):
        """初始化可视化器
        
        Args:
            log_dir: 图表保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化记录列表
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.gpu_memories = []
        self.epochs = []
    
    def update_plots(self, epoch: int, train_loss: float, val_loss: float,
                     train_acc: float, val_acc: float, lr: float,
                     gpu_memory: Optional[float] = None):
        """更新训练指标的可视化
        
        Args:
            epoch: 当前训练轮数
            train_loss: 训练损失
            val_loss: 验证损失
            train_acc: 训练准确率
            val_acc: 验证准确率
            lr: 当前学习率
            gpu_memory: GPU内存使用量（可选）
        """
        # 记录数据
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
        if gpu_memory is not None:
            self.gpu_memories.append(gpu_memory)
        
        # 在控制台输出当前指标
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {lr:.6f}')
        if gpu_memory is not None:
            print(f'GPU Memory: {gpu_memory:.2f} MB')
    
    def close(self):
        """保存训练过程的可视化图表"""
        # 设置图表样式
        plt.style.use('seaborn')
        
        # 创建并保存损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'loss_curve.png'))
        plt.close()
        
        # 创建并保存准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_accs, label='Train Accuracy')
        plt.plot(self.epochs, self.val_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'accuracy_curve.png'))
        plt.close()
        
        # 创建并保存学习率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.learning_rates)
        plt.title('Learning Rate over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'learning_rate_curve.png'))
        plt.close()
        
        # 如果有GPU内存数据，创建并保存内存使用曲线
        if self.gpu_memories:
            plt.figure(figsize=(10, 6))
            plt.plot(self.epochs, self.gpu_memories)
            plt.title('GPU Memory Usage over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Memory Usage (MB)')
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, 'gpu_memory_curve.png'))
            plt.close()