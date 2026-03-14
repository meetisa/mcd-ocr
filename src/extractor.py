import pytesseract
import numpy as np
from typing import List
from .models import OCRWord, BoundingBox


class TesseractExtractor:
    def __init__(self, lang: str = "ita") -> None:
        self.lang = lang
        self.config = "--psm 6"

    def extract(self, processed_image: np.ndarray) -> List[OCRWord]:
        data = pytesseract.image_to_data(
            processed_image,
            lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT,
        )

        words: List[OCRWord] = []

        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            if text and conf > 0:
                geometry = BoundingBox(
                    x=data["left"][i],
                    y=data["top"][i],
                    w=data["width"][i],
                    h=data["height"][i],
                )
                words.append(OCRWord(text=text, confidence=conf, geometry=geometry))

        return words
