import pytest
import numpy as np
import cv2
from src import processor
from src.processor import ImageProcessor


def test_processor_returns_binary_matrix(tmp_path):
    test_img_path = tmp_path / "test.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(test_img_path), img)

    processor = ImageProcessor()
    result = processor.process(test_img_path)

    assert len(result.shape) == 2
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8
