import pytest
import os
import sys
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.common.predict import predict_folder

def test_predict_folder_cnn():
    """测试使用 CNN 的预测"""
    input_dir = "test_input"
    model_path = "models/mobilenetv3_top.pth"

    if not os.path.exists(input_dir):
        pytest.skip("测试数据不存在，请先运行 prepare_test.py")
    if not os.path.exists(model_path):
        pytest.skip("CNN 模型不存在")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = os.path.join(tmpdir, "result_cnn.csv")

        results = predict_folder(input_dir, output_csv, model_path=model_path, use_cnn=True)

        assert len(results) == 10
        for filename, angle in results:
            assert 0 <= angle <= 80
            assert isinstance(angle, float)

        assert os.path.exists(output_csv)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
