import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ValveAngleRegressor(nn.Module):
    """阀门角度回归模型"""

    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        # 替换最后的分类层
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

def get_transform():
    """获取图片预处理"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_cnn(image_path: str, model_path: str) -> float:
    """
    使用 CNN 预测阀门开度角度

    Args:
        image_path: 图片路径
        model_path: 模型路径

    Returns:
        预测角度 (0-80)
    """
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ValveAngleRegressor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 加载图片
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 3. 预测
    with torch.no_grad():
        output = model(image_tensor)
        angle = output.item()

    # 4. 限制范围
    angle = min(max(angle, 0.0), 80.0)

    return round(angle, 1)
