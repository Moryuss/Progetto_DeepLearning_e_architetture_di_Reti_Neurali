import os
import cv2
import numpy as np
import torch
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import recognize_faces, load_dataset_embeddings
from facenet_pytorch import InceptionResnetV1


def main():
    # Config
    dataset_dir = "data/dataset"
    yolo_model_path = "models/face_detection/yolo11_nano.pt"
    recognizer_model_path = "models/face_recognition/vggface2.pt"

    # Inizializza detector e recognizer
    detector = FaceDetector(model_path=yolo_model_path)
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone, model_path=recognizer_model_path)

    # Carica embeddings del dataset
    embeddings_array, labels_list = load_dataset_embeddings(dataset_dir)

    # Webcam
    cap = cv2.VideoCapture(0)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
