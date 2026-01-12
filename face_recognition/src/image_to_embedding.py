import os
import json
import numpy as np
import cv2
import torch
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.inizializer import initialization_detector_recognizer
from src.utils import get_model_name
from src.config import (
    DATASET_DIR,
    DETECTOR_MODEL_PATH,
    RECOGNIZER_MODEL_PATH,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def process_dataset(dataset_dir: str,
                    detector: FaceDetector,
                    recognizer: FaceRecognizer,
                    model_name: str):
    dataset_dir = os.path.abspath(dataset_dir)

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        images_dir = os.path.join(person_path, "images")
        augmented_dir = os.path.join(person_path, "augmented")

        # Path con nome modello
        embeddings_path = os.path.join(
            person_path, f"embeddings_{model_name}.npz")
        metadata_path = os.path.join(
            person_path, f"metadata_{model_name}.json")

        if not os.path.exists(images_dir):
            print(f"[SKIP] {person_name}: manca images/")
            continue

        image_files = []
        for folder, tag in [(images_dir, "original"), (augmented_dir, "augmented")]:
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    image_files.append((os.path.join(folder, fname), tag))

        if not image_files:
            print(f"[WARN] {person_name}: nessuna immagine trovata")
            continue

        embeddings = {}
        metadata = {
            "person": person_name,
            "model_name": model_name,
            "total_images": 0,
            "original_images": 0,
            "augmented_images": 0,
            "skipped_images": []
        }

        for img_path, img_type in image_files:
            fname = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                metadata["skipped_images"].append(fname)
                continue

            faces = detector.detect(img)
            if len(faces) != 1:
                print(
                    f"[WARN] {person_name}/{fname}: {len(faces)} volti trovati, skip")
                metadata["skipped_images"].append(fname)
                continue

            x1, y1, x2, y2 = faces[0]["bbox"]
            face_crop = img[y1:y2, x1:x2]

            emb = recognizer.get_embedding(face_crop)
            embeddings[fname] = emb
            metadata["total_images"] += 1

            if img_type == "original":
                metadata["original_images"] += 1
            else:
                metadata["augmented_images"] += 1

        if embeddings:
            np.savez_compressed(embeddings_path, **embeddings)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(
                f"[OK] {person_name}: "
                f"{metadata['total_images']} embeddings "
                f"({metadata['original_images']} orig, "
                f"{metadata['augmented_images']} aug)"
            )
        else:
            print(f"[WARN] {person_name}: nessun embedding salvato")


def main():
    # se danno errore --> str(...)
    dataset_dir = str(DATASET_DIR)
    detector_model_path = str(DETECTOR_MODEL_PATH)
    recognizer_model_path = str(RECOGNIZER_MODEL_PATH)

    detector, recognizer = initialization_detector_recognizer(
        detector_model_path, recognizer_model_path)

    # Estrae nome modello
    model_name = get_model_name(recognizer)
    print(f"[INFO] Using model: {model_name}")

    process_dataset(str(dataset_dir), detector, recognizer, model_name)

    detector.close()


if __name__ == "__main__":
    main()
