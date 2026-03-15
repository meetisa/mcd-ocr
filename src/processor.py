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

    def _grid_mask(self, image_path: Path) -> np.ndarray:
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
        return grid_mask
        """
        red_img = np.zeros_like(img_color)
        red_img[:] = (0, 0, 255)

        debug_view = np.where(grid_mask[..., None] > 0, red_img, img_color)

        return debug_view
        """

    def slice_cells(self, image_path: Path):
        img = cv2.imread(str(image_path))
        grid_mask = self._grid_mask(image_path)

        contours, _ = cv2.findContours(
            grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 30 < w < 600 and 15 < h < 150:
                cells.append((x, y, w, h))

        cells.sort(key=lambda c: (c[1] // 20, c[0]))
        return cells, img

    def text_slice_row(self, image_path: Path, target_row_index: int):
        cells, img = self.slice_cells(image_path)

        rows = []
        if cells:
            current_row = [cells[0]]
            for i in range(1, len(cells)):
                if abs(cells[i][1] - current_row[-1][1]) < 20:
                    current_row.append(cells[i])
                else:
                    rows.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [cells[i]]
            rows.append(sorted(current_row, key=lambda c: c[0]))

        if target_row_index < len(rows):
            my_row = rows[target_row_index]
            output_dir = Path("/data_output/row_17_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[+] Estrazione riga {target_row_index}: trovate {len(my_row)} celle"
            )
            for idx, (x, y, w, h) in enumerate(my_row):
                cell_img = img[y : y + h, x : x + w]
                cv2.imwrite(str(output_dir / f"cell_{idx}.png"), cell_img)
            return True
        else:
            print(f"[-] Errore: la tabella sembra avere solo {len(rows)} righe")
            return False

    def debug_all_cells(self, image_path: Path):
        cells, img = self.slice_cells(image_path)

        debug_img = img.copy()
        for idx, (x, y, w, h) in enumerate(cells):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                str(idx),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        output_path = Path("/data_output/full_grid_debug.png")
        cv2.imwrite(str(output_path), debug_img)
        return output_path
