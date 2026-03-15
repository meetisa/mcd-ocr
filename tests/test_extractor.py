import pytest
import numpy as np
import cv2
from src.extractor import TesseractExtractor
from src.models import OCRWord


def test_extractor_identifies_synthetic_text():
    img = np.full((200, 500), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "ROSSI", (50, 100), font, 2, (0), 3, cv2.LINE_AA)

    extractor = TesseractExtractor(lang="ita")
    results = extractor.extract(img)

    assert len(results) > 0

    found_words = [w.text.upper() for w in results]
    assert "ROSSI" in found_words

    for word in results:
        if word.text.upper() == "ROSSI":
            assert word.confidence > 50
            assert word.geometry.w > 0
            assert word.geometry.h > 0
