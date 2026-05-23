import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cv_predictor import predict_cv

def test_predict_cv_with_valid_image():
    """测试有效图片的预测"""
    # 使用 origin data/top/ 中的第一张图片
    image_path = "origin data/top/0001_3.4.jpg"
    if os.path.exists(image_path):
        angle = predict_cv(image_path)
        assert 0 <= angle <= 80
        assert isinstance(angle, float)

def test_predict_cv_invalid_path():
    """测试无效路径"""
    with pytest.raises(ValueError):
        predict_cv("nonexistent.jpg")
