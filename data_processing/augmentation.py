from torchvision import transforms
from utils.config import AUGMENTATION_CONFIG, NORMALIZATION_CONFIG

def get_train_transform():
    """获取训练数据增强转换"""
    return transforms.Compose([
        transforms.RandomResizedCrop(AUGMENTATION_CONFIG['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(AUGMENTATION_CONFIG['rotation_degrees']),
        transforms.ColorJitter(
            brightness=AUGMENTATION_CONFIG['brightness'],
            contrast=AUGMENTATION_CONFIG['contrast'],
            saturation=AUGMENTATION_CONFIG['saturation'],
            hue=AUGMENTATION_CONFIG['hue']
        ),
        transforms.RandomAffine(degrees=AUGMENTATION_CONFIG['rotation_degrees'], scale=(0.8, 1.2)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_CONFIG['mean'],
                         std=NORMALIZATION_CONFIG['std'])
    ])

def get_eval_transform():
    """获取验证和测试数据预处理转换"""
    return transforms.Compose([
        transforms.Resize(AUGMENTATION_CONFIG['resize_size']),
        transforms.CenterCrop(AUGMENTATION_CONFIG['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_CONFIG['mean'],
                         std=NORMALIZATION_CONFIG['std'])
    ])