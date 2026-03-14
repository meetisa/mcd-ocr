import re
from pathlib import Path
from typing import Optional
from datetime import datetime
from .models import ImageMetadata


class FileDiscovery:
    def __init__(self, directory: Path):
        self.directory = directory
        self.pattern = re.compile(r"(\d{8})_(\d{6})")

    def _parse_timestamp(self, filename: str) -> Optional[datetime]:
        match = self.pattern.search(filename)
        if match:
            try:
                return datetime.strptime(
                    f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S"
                )
            except ValueError:
                return None
        return None

    def get_latest(self) -> ImageMetadata:
        files = [f for f in self.directory.glob("*.jpg")]
        if not files:
            raise FileNotFoundError(f"Nessun file .jpg trovato in {self.directory}")

        latest_path = sorted(files)[-1]
        timestamp = self._parse_timestamp(latest_path.name) or datetime.fromtimestamp(0)

        return ImageMetadata(
            path=latest_path, timestamp=timestamp, raw_name=latest_path.name
        )
