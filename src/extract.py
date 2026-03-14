import cv2
import pytesseract
import os
import datetime
import numpy as np

INPUT_DIR = "/data_input"
OUTPUT_DIR = "/data_output"
FILTERED_DIR = "/app/filtered"

CREW = "AMADUCCI" #os.getenv("CREW", "AMADUCCI SAMUELE")

def get_latest_image(dir):
    valid_extensions = ('.jpg')
    files = [f for f in os.listdir(dir) if f.lower().endswith(valid_extensions)]

    if not files:
        return None

    files.sort()
    latest_filename = files[-1]
    return os.path.join(dir, latest_filename)


def process_schedule():
    image_path = get_latest_image(INPUT_DIR)
    if not image_path:
        print(f"[-] Errore: Nessuna immagine trovata in {INPUT_DIR}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[-] Errore: Impossibile leggere l'immagine {image_path}")
        return

    print(f"[*] Elaborazione di: {os.path.basename(image_path)}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    d = pytesseract.image_to_data(thresh, lang='ita', output_type='dict')

    n_boxes = len(d['text'])
    target_y = -1
    target_h = 0

    for i in range(n_boxes):
        text = d['text'][i].strip().upper()
        conf = int(d['conf'][i])

        if len(text) >= 3 and conf > 50:
            if text in CREW or CREW in text:
                target_y = d['top'][i]
                target_h = d['height'][i]
            
                print(f"[+] Match trovato: '{text}' a coordinata Y: {target_y}") 
                cv2.rectangle(
                    img,
                    (d['left'][i], d['top'][i]),
                    (d['left'][i]+d['width'][i], d['top'][i]+d['height'][i]),
                    (0, 255, 0),
                    2
                )
                break

    if target_y != -1:
        y_start = max(0, target_y + 15)
        y_end = target_y + target_h + 15

        roi = thresh[y_start:y_end, 0:img.shape[1]]

        final_text = pytesseract.image_to_string(roi, lang='ita').strip()

        output_file = os.path.join(OUTPUT_DIR, "turno.txt")
        with open(output_file, "w") as f:
            f.write(f"File analizzato: {os.path.basename(image_path)}`\n")
            f.write(f"Turno estratto il {datetime.datetime.now()}:\n")
            f.write("-" * 30 + "\n")
            f.write(final_text)

        cv2.line(img, (0, target_y), (img.shape[1], target_y), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_view.png"), img)
        print(f"[!] Risultato salvato in {OUTPUT_DIR}")

    else:
        print(f"[-] Nome '{CREW}' non trovato")


if __name__ == "__main__":
    process_schedule()
