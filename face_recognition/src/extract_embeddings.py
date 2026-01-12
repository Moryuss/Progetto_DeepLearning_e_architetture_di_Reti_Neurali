import os
import json
import numpy as np
import cv2
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.inizializer import initialization_detector_recognizer

from src.config import (
    PEOPLE_DIR,
    KNOWN_PEOPLE_DIR,
    PEOPLE_EMB_PATH,
    KNOWN_EMB_PATH,
    DETECTOR_MODEL_PATH,
    RECOGNIZER_MODEL_PATH

)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_embeddings(
    images_dir: str,
    detector,
    recognizer,
    output_path: str,
    metadata_path: str
):
    embeddings = {}
    metadata = {
        "total_images": 0,
        "skipped_images": []
    }

    images_dir = os.path.abspath(images_dir)

    for fname in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            metadata["skipped_images"].append(fname)
            continue

        # estrae faccia
        faces = detector.detect(img)

        if len(faces) != 1:
            metadata["skipped_images"].append(fname)
            continue

        x1, y1, x2, y2 = faces[0]["bbox"]
        face_crop = img[y1:y2, x1:x2]

        # estrae embedding
        emb = recognizer.get_embedding(face_crop)

        embeddings[fname] = emb
        metadata["total_images"] += 1

    if embeddings:
        np.savez_compressed(output_path, **embeddings)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Saved {len(embeddings)} embeddings → {output_path}")
    else:
        print("[WARN] No embeddings saved")


def main():
    # se danno errore --> str(...)
    people_dir = str(PEOPLE_DIR)
    known_people_dir = str(KNOWN_PEOPLE_DIR)
    detector_model_path = str(DETECTOR_MODEL_PATH)
    recognizer_model_path = str(RECOGNIZER_MODEL_PATH)
    people_embeddings_path = str(PEOPLE_EMB_PATH)
    known_embeddings_path = str(KNOWN_EMB_PATH)

    detector, recognizer = initialization_detector_recognizer(
        detector_model_path, recognizer_model_path)

    # embedding di persone note e ignote
    # NOTA se non vuoi runnare quello delle immagini di people che sono molte, commentalo easy
   # extract_embeddings(people_dir, detector, recognizer,
   #                    people_embeddings_path, "metadata_people.json")
    extract_embeddings(known_people_dir, detector, recognizer,
                       known_embeddings_path, "metadata_known.json")

    detector.close()


if __name__ == "__main__":
    main()
