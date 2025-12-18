import os
import cv2
import numpy as np
import torch
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import recognize_faces, load_dataset_embeddings, draw_label
from facenet_pytorch import InceptionResnetV1

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
    detector = FaceDetector(model_path=yolo_model_path)
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone, model_path=recognizer_model_path)

    # Carica embeddings del dataset
    embeddings_array, labels_list = load_dataset_embeddings(dataset_dir)

    # Webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 640, 480)

    if not cap.isOpened():
        print("Errore nell'aprire la webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = recognize_faces(
            frame, detector, recognizer, embeddings_array, labels_list)

        # Disegna bounding box
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            name = r["name"]
            confidence = r["confidence"]
            draw_label(frame, name, confidence, (x1, y1, x2, y2),
                       font_scale=1, color=(0, 255, 0), thickness=2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
