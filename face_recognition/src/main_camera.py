import os
import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
from src.utils import recognize_faces, load_dataset_embeddings, draw_label
from src.inizializer import initialization_detector_recognizer

from src.config import (
    DATASET_DIR,
    DETECTOR_MODEL_PATH,
    RECOGNIZER_MODEL_PATH,
    CLASSIFY_IMAGES_DIR,
    DEFAULT_MODEL,
    EMBEDDINGS_DIR,
    get_model_config
)
import argparse


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Face Recognition from Camera')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Recognition model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.60,
        help='Recognition threshold (default: 0.60)'
    )
    return parser.parse_args()


def main():
    # Config
    args = parse_args()
    model_name = args.model
    dataset_dir = str(DATASET_DIR)

    yolo_model_path = str(DETECTOR_MODEL_PATH)
    recognizer_model_path = str(RECOGNIZER_MODEL_PATH)
    # cartella con immagini da classificare
    images_dir = str(CLASSIFY_IMAGES_DIR)

    # Inizializza detector e recognizer
    detector, recognizer = initialization_detector_recognizer(
        yolo_model_path, recognizer_model_path,         model_name=model_name)

    # Carica embeddings del dataset
    embeddings_array, labels_list = load_dataset_embeddings(
        dataset_dir, recognizer=recognizer,         model_name=model_name)

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

        cv2.putText(frame, f"Model: {model_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
