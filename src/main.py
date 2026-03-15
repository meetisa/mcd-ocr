from pathlib import Path
import cv2
from src.discovery import FileDiscovery
from src.processor import ImageProcessor
from src.extractor import TesseractExtractor


def main():
    discovery = FileDiscovery(Path("/data_input"))
    latest = discovery.get_latest()

    print(f"[+] Trovato il file {latest.raw_name}")

    processor = ImageProcessor()
    processed_matrix = processor.process(latest.path, apply_morphology=True)

    output_path = Path("/data_output/test_semantics.png")
    cv2.imwrite(str(output_path), processed_matrix)

    print(f"[+] Immagine processata salvata in: {output_path}")
    print(f"[+] Risoluzione: {processed_matrix.shape}")

    print(f"[+] Verificando struttura tabella su: {latest.raw_name}")
    table_debug_img = processor.test_table_detection(latest.path)
    output_path = Path("/data_output/table_structure_debug.png")
    cv2.imwrite(str(output_path), table_debug_img)
    print(f"[+] Vista di debug salvata in: {output_path}")

    """
    extractor = TesseractExtractor()
    words = extractor.extract(processed_matrix)

    print(f"[+] Estrazione completata: trovate {len(words)} parole")
    for word in words:
        print(f"    - {word.text} (Conf: {word.confidence}%) at Y: {word.geometry.y}")
    """


if __name__ == "__main__":
    main()
