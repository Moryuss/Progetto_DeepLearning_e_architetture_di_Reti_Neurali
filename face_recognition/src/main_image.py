import os
import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from src.utils import draw_label, recognize_faces, load_dataset_embeddings, resize_max, draw_label
from src.inizializer import initialization_detector_recognizer
from src.config import (
    DATASET_DIR,
    DETECTOR_MODEL_PATH,
    RECOGNIZER_MODEL_PATH,
    CLASSIFY_IMAGES_DIR,
    EMBEDDINGS_DIR
)


def main():
    # Config
    dataset_dir = str(DATASET_DIR)
    yolo_model_path = str(DETECTOR_MODEL_PATH)
    recognizer_model_path = str(RECOGNIZER_MODEL_PATH)
    # cartella con immagini da classificare
    images_dir = str(CLASSIFY_IMAGES_DIR)

    # Inizializza detector e recognizer
    detector, recognizer = initialization_detector_recognizer(
        yolo_model_path, recognizer_model_path)

    # Carica embeddings del dataset
    embeddings_array, labels_list = load_dataset_embeddings(dataset_dir)

    # Scansiona tutte le immagini nella cartella
    for fname in os.listdir(images_dir):
        img_path = os.path.join(images_dir, fname)
        if not os.path.isfile(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Impossibile leggere {img_path}")
            continue

        results = recognize_faces(
            frame, detector, recognizer, embeddings_array, labels_list)

        # Disegna bounding box e nome
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            name = r["name"]
            confidence = r["confidence"]

            draw_label(frame, name, confidence, (x1, y1, x2, y2),
                       font_scale=3, thickness=4)

        # Mostra immagine classificata
        frame = resize_max(frame, max_dim=800)
        cv2.imshow(f"Classified: {fname}", frame)
        cv2.waitKey(0)  # premi un tasto per passare all'immagine successiva
        cv2.destroyAllWindows()

    detector.close()


if __name__ == "__main__":
    main()
