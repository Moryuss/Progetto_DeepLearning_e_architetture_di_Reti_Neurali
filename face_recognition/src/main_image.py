import os
import cv2
import numpy as np
import torch
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import draw_label, recognize_faces, load_dataset_embeddings, resize_max, draw_label
from facenet_pytorch import InceptionResnetV1


def main():
    # Config
    dataset_dir = "data/dataset"
    yolo_model_path = "models/face_detection/yolo11_nano.pt"
    recognizer_model_path = "models/face_recognition/vggface2.pt"
    images_dir = "data/classify_images"  # cartella con immagini da classificare
    # images_dir = "data/dataset/Delia_Confortini/augmented"

    # Inizializza detector e recognizer
    detector = FaceDetector(model_path=yolo_model_path)
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone, model_path=recognizer_model_path)

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
