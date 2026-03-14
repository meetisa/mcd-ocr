from pathlib import Path
import cv2
from src.discovery import FileDiscovery
from src.processor import ImageProcessor


def main():
    discovery = FileDiscovery(Path("/data_input"))
    latest = discovery.get_latest()

    print(f"[+] Trovato il file {latest.raw_name}")

    processor = ImageProcessor()
    processed_matrix = processor.process(latest.path)

    output_path = Path("/data_output/test_semantics.png")
    cv2.imwrite(str(output_path), processed_matrix)

    print(f"[+] Immagine processata salvata in: {output_path}")
    print(f"[+] Risoluzione: {processed_matrix.shape}")


if __name__ == "__main__":
    main()
