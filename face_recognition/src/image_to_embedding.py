import os
import json
import numpy as np
import cv2
import torch

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def process_dataset(dataset_dir: str,
                    detector: FaceDetector,
                    recognizer: FaceRecognizer):

    dataset_dir = os.path.abspath(dataset_dir)

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        images_dir = os.path.join(person_path, "images")
        augmented_dir = os.path.join(person_path, "augmented")

        embeddings_path = os.path.join(person_path, "embeddings.npz")
        metadata_path = os.path.join(person_path, "metadata.json")

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

            face_tensor = recognizer._preprocess(face_crop)

            with torch.no_grad():
                emb = recognizer.model(face_tensor)
                emb = emb.cpu().numpy().flatten()

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
    dataset_dir = "data/dataset"
    yolo_model_path = "models/face_detection/yolo11_nano.pt"
    recognizer_model_path = "models/face_recognition/vggface2.pt"

    detector = FaceDetector(model_path=yolo_model_path)

    from facenet_pytorch import InceptionResnetV1
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone,
        model_path=recognizer_model_path
    )

    process_dataset(dataset_dir, detector, recognizer)
    detector.close()


if __name__ == "__main__":
    main()
