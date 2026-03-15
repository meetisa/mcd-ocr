import cv2
import numpy as np
from pathlib import Path
from typing import Protocol


class Processor(Protocol):
    def process(self, image_path: Path) -> np.ndarray: ...


class ImageProcessor:
    def __init__(self, kernel_size: int = 2):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.line_scale = 40

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _apply_adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20
        )

    def process(self, image_path: Path, apply_morphology: bool = False) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise IOError(f"Impossibile leggere l'immagine al percorso {image_path}")

        height, width = img.shape[:2]
        img = cv2.resize(img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

        gray = self._to_grayscale(img)
        binary = self._apply_adaptive_threshold(gray)

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.erode(binary, kernel, iterations=1)
        return dilated

        if not apply_morphology:
            return binary

        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        return cleaned

    def test_table_detection(self, image_path: Path) -> np.ndarray:
        img_color = cv2.imread(str(image_path))
        if img_color is None:
            raise IOError(f"Impossibile leggere: {image_path}")

        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        width = binary.shape[1]
        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (width // self.line_scale, 1)
        )
        horiz_mask = cv2.erode(binary, horiz_kernel, iterations=1)
        horiz_mask = cv2.dilate(horiz_mask, horiz_kernel, iterations=1)

        height = binary.shape[0]
        vert_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, height // self.line_scale)
        )
        vert_mask = cv2.erode(binary, vert_kernel, iterations=1)
        vert_mask = cv2.dilate(vert_mask, vert_kernel, iterations=1)

        grid_mask = cv2.add(horiz_mask, vert_mask)

        red_img = np.zeros_like(img_color)
        red_img[:] = (0, 0, 255)

        debug_view = np.where(grid_mask[..., None] > 0, red_img, img_color)

        return debug_view
