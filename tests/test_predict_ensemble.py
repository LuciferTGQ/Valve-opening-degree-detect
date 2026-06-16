import pytest
import os
import sys
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.common.ensemble import predict_ensemble

def test_predict_ensemble():
    """测试融合预测（CV + CNN）"""
    input_dir = "test_input"
    model_path = "models/mobilenetv3_top.pth"

    if not os.path.exists(input_dir):
        pytest.skip("测试数据不存在，请先运行 prepare_test.py")
    if not os.path.exists(model_path):
        pytest.skip("CNN 模型不存在")

    images = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])

    for img in images:
        img_path = os.path.join(input_dir, img)
        angle = predict_ensemble(img_path, model_path)
        assert 0 <= angle <= 80
        assert isinstance(angle, float)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
