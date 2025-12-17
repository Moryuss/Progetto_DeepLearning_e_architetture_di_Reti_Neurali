import os
import json
import numpy as np
import cv2
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.utils import preprocess_for_recognizer
import torch


def process_dataset(dataset_dir: str, detector: FaceDetector, recognizer: FaceRecognizer):
    """
    Processa un dataset di immagini per persona:
    - rileva volti
    - calcola embedding
    - salva embeddings in .npz
    - aggiorna metadata.json
    """
    dataset_dir = os.path.abspath(dataset_dir)
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        images_path = os.path.join(person_path, "images")
        embeddings_path = os.path.join(person_path, "embeddings.npz")
        metadata_path = os.path.join(person_path, "metadata.json")

        embeddings = {}
        metadata = {"num_images": 0, "image_names": []}

        if not os.path.exists(images_path):
            print(
                f"[WARN] Nessuna cartella 'images' per {person_name}, salto...")
            continue

        for fname in os.listdir(images_path):
            img_path = os.path.join(images_path, fname)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Impossibile leggere {img_path}")
                continue

            # Rileva volti
            faces = detector.detect(img)
            if len(faces) == 0:
                print(f"[WARN] Nessun volto trovato in {fname}")
                continue

            # Prendi il primo volto rilevato. Nota che non funziona se ci sono più volti da distinguere, devi "cropparli" in modod che sia uno solo
            x1, y1, x2, y2 = faces[0]['bbox']

            face_crop = img[y1:y2, x1:x2]

            # Preprocess e calcola embedding
            face_tensor = recognizer._preprocess(face_crop)
            with torch.no_grad():
                embedding = recognizer.model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()

            embeddings[fname] = embedding.tolist()
            metadata["num_images"] += 1
            metadata["image_names"].append(fname)

        # Salva embeddings
        if embeddings:
            np.savez_compressed(embeddings_path, **embeddings)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(
                f"[INFO] Processati {metadata['num_images']} immagini per {person_name}")
        else:
            print(f"[WARN] Nessun embedding salvato per {person_name}")


def main():
    # Modifica questi percorsi secondo il tuo setup
    dataset_dir = "data/dataset"
    yolo_model_path = "models/face_detection/yolo11_nano.pt"
    recognizer_model_path = "models/face_recognition/vggface2.pt"

    # Inizializza detector e recognizer
    detector = FaceDetector(model_path=yolo_model_path)

    from facenet_pytorch import InceptionResnetV1
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone, model_path=recognizer_model_path)

    process_dataset(dataset_dir, detector, recognizer)

    # Chiudi detector
    detector.close()


if __name__ == "__main__":
    main()
