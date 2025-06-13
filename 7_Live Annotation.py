import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time
import cv2
from ultralytics import YOLO

# ========================== Einstellungen ==========================
MODEL_PATH = r"G:\Everyone\TEMP\465\INSPECTUBE\runs\train\yolo11n_custom\weights\best.pt" # Modellpfad
IMG_DIR = r"C:\Users\hagmmart\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\23.05.25_M_Innder_B_Dataset_INSPECTUBE\3_Splitted\test"   # Ordner mit Bildern
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")  # Bildtypen

# ========== Bilder sammeln =========
image_paths = [f for f in glob.glob(os.path.join(IMG_DIR, "*")) if f.lower().endswith(IMG_EXTENSIONS)]
image_paths.sort()
if not image_paths:
    print("Keine Bilder gefunden!")
    exit(1)

print(f"{len(image_paths)} Bilder gefunden.")

# ========== Modell laden ===========
model = YOLO(MODEL_PATH)
names = model.names

# ========== Galerie-Loop ===========
idx = 0
while True:
    img_path = image_paths[idx]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Konnte Bild nicht laden: {img_path}")
        continue

    t0 = time.time()
    results = model(img, verbose=False)
    t1 = time.time()
    infer_time = t1 - t0

    res = results[0]
    boxes = res.boxes
    out_img = img.copy()

    # Zeichne BBoxen und Labels
    if boxes is not None and boxes.shape[0] > 0:
        for box, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[cls]} {conf:.2f}"
            color = (0, 200, 0)
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out_img, label, (x1, max(y1-8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Titel mit Bildname und Zeit
    title = f"[{idx+1}/{len(image_paths)}] Inferenzzeit: {infer_time*1000:.1f} ms"
    out_img_disp = out_img.copy()
    cv2.putText(out_img_disp, title, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    # Zeige Bild gross
    cv2.imshow("YOLO Gallerie", out_img_disp)
    key = cv2.waitKey(0)  # Warte auf Tastendruck

    # Steuerung: rechts/links vor/zurück, ESC = Ende
    if key in [27, ord('q')]:  # ESC oder q zum Beenden
        break
    elif key == 81 or key == ord('a'):  # links (Pfeil links oder a)
        idx = (idx - 1) % len(image_paths)
    elif key == 83 or key == ord('d'):  # rechts (Pfeil rechts oder d)
        idx = (idx + 1) % len(image_paths)
    else:
        idx = (idx + 1) % len(image_paths)  # Standard: vorwärts

    cv2.destroyWindow("YOLO Gallerie")

cv2.destroyAllWindows()
print("Galerie beendet.")

