from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from datetime import datetime

@dataclass(frozen=True)
class ImageMetadata:
    path: Path
    timestamp: datetime
    raw_name: str

@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class OCRWord:
    text: str
    confidence: float
    geometry: BoundingBox
