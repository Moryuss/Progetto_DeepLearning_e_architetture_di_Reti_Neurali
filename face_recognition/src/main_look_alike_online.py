import os
import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from src.inizializer import initialization_detector_recognizer
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import load_embeddings, draw_label, load_image_safe, find_top_k

from src.config import (
    PEOPLE_EMB_PATH,
    DETECTOR_MODEL_PATH,
    RECOGNIZER_MODEL_PATH,
    PEOPLE_DIR
)

# ---------------- CONFIG ----------------
TOP_K = 1          # mostra solo il match migliore
SIM_THRESHOLD = 0.65
MAX_IMG_SIZE = 300
# ----------------------------------------


def main():
    yolo_model_path = str(DETECTOR_MODEL_PATH)
    recognizer_model_path = str(RECOGNIZER_MODEL_PATH)

    # cartella con immagini da classificare
    # ---- carica PEOPLE embeddings ----
    people_embs, people_names = load_embeddings(PEOPLE_EMB_PATH)
    print(f"[INFO] PEOPLE embeddings: {len(people_names)}")

    # Inizializza detector e recognizer
    detector, recognizer = initialization_detector_recognizer(
        yolo_model_path, recognizer_model_path)

    # ---- apri webcam ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] impossibile aprire la webcam")
        return

    cv2.namedWindow("Live Similarity", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Similarity", 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_crop = frame[y1:y2, x1:x2]

            emb = recognizer.get_embedding(face_crop)

            top_match = find_top_k(emb, people_embs, people_names, TOP_K)[0]
            match_name, match_score = top_match

            if match_score >= SIM_THRESHOLD:
                label = f"{match_name} ({match_score:.2f})"
            else:
                label = f"No strong match ({match_score:.2f})"

            # bounding box e label sul frame
            draw_label(frame, label, match_score, (x1, y1, x2, y2),
                       font_scale=1, thickness=2, color=(0, 255, 0))

            # mostra immagine più simile a lato
            match_img = load_image_safe(os.path.join(
                PEOPLE_DIR, match_name), MAX_IMG_SIZE)
            if match_img is not None:
                cv2.putText(match_img, f"{match_score:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # concat verticale
                combined = cv2.hconcat([cv2.resize(frame, (MAX_IMG_SIZE, MAX_IMG_SIZE)),
                                        cv2.resize(match_img, (MAX_IMG_SIZE, MAX_IMG_SIZE))])
                cv2.imshow("Live Similarity", combined)
            else:
                cv2.imshow("Live Similarity", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
