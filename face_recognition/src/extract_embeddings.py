import os
import json
import numpy as np
import cv2
import argparse
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.inizializer import initialization_detector_recognizer
from src.config import (
    PEOPLE_DIR,
    KNOWN_PEOPLE_DIR,
    DETECTOR_MODEL_PATH,
    DEFAULT_MODEL,
    get_model_config,
    get_people_embeddings_path,
    get_known_embeddings_path
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_embeddings(
    images_dir: str,
    detector,
    recognizer,
    output_path: str,
    metadata_path: str,
    model_name: str,
    suffix: str
):
    embeddings = {}
    metadata = {
        "model_name": model_name,
        "model_suffix": suffix,
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

        faces = detector.detect(img)
        if len(faces) != 1:
            metadata["skipped_images"].append(fname)
            continue

        x1, y1, x2, y2 = faces[0]["bbox"]
        face_crop = img[y1:y2, x1:x2]

        emb = recognizer.get_embedding(face_crop)
        embeddings[fname] = emb
        metadata["total_images"] += 1

    if embeddings:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **embeddings)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"OK Saved {len(embeddings)} embeddings -> {output_path}")
    else:
        print("WARN No embeddings saved")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract embeddings for look-alike')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Model to use for embeddings')
    parser.add_argument('--skip-people', action='store_true',
                        help='Skip people embeddings extraction')
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model

    print(f"Starting embedding extraction for look-alike")
    print(f"Model: {model_name}")

    model_config = get_model_config(model_name)
    suffix = model_config['embeddings_suffix']

    people_dir = str(PEOPLE_DIR)
    known_people_dir = str(KNOWN_PEOPLE_DIR)
    detector_model_path = str(DETECTOR_MODEL_PATH)

    detector, recognizer = initialization_detector_recognizer(
        detector_model_path,
        model_name=model_name
    )

    people_embeddings_path = get_people_embeddings_path(model_name)
    known_embeddings_path = get_known_embeddings_path(model_name)

    people_metadata_path = people_embeddings_path.parent / \
        f"metadata_people_{suffix}.json"
    known_metadata_path = known_embeddings_path.parent / \
        f"metadata_known_{suffix}.json"

    if not args.skip_people:
        print("\nExtracting PEOPLE embeddings...")
        extract_embeddings(
            people_dir,
            detector,
            recognizer,
            people_embeddings_path,
            people_metadata_path,
            model_name,
            suffix
        )
    else:
        print("\nSkipping PEOPLE embeddings (--skip-people flag)")

    print("\nExtracting KNOWN PEOPLE embeddings...")
    extract_embeddings(
        known_people_dir,
        detector,
        recognizer,
        known_embeddings_path,
        known_metadata_path,
        model_name,
        suffix
    )

    detector.close()
    recognizer.close()

    print("\n=== EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()
