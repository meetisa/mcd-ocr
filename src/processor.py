import cv2
import numpy as np
from pathlib import Path
from typing import Protocol


class Processor(Protocol):
    def process(self, image_path: Path) -> np.ndarray: ...


class ImageProcessor:
    def __init__(self, kernel_size: int = 2):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _apply_adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def process(self, image_path: Path, apply_morphology: bool = False) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise IOError(f"Impossibile leggere l'immagine al percorso {image_path}")

        gray = self._to_grayscale(img)
        binary = self._apply_adaptive_threshold(gray)

        if not apply_morphology:
            return binary

        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        return cleaned
