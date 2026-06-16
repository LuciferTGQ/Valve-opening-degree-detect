import pytest
import os
import sys
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.common.predict import predict_folder

def test_predict_folder_cv_only():
    """测试仅使用 CV 的预测"""
    # 使用 origin data/top 中的前 10 张图片
    input_dir = "origin data/top"

    if not os.path.exists(input_dir):
        pytest.skip("测试数据不存在")

    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = os.path.join(tmpdir, "result.csv")

        # 获取前 10 张图片
        images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])[:10]

        # 创建测试文件夹
        test_dir = os.path.join(tmpdir, "test_images")
        os.makedirs(test_dir)
        for img in images:
            shutil.copy(os.path.join(input_dir, img), test_dir)

        # 运行预测
        results = predict_folder(test_dir, output_csv, use_cnn=False)

        # 验证
        assert len(results) == 10
        for filename, angle in results:
            assert 0 <= angle <= 80
            assert isinstance(angle, float)

        # 验证 CSV 文件
        assert os.path.exists(output_csv)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
