import os
import json
import hashlib
import numpy as np
import cv2
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.inizializer import initialization_detector_recognizer
from src.config import (
    DATASET_DIR,
    DETECTOR_MODEL_PATH,
    DEFAULT_MODEL,
    get_model_config
)
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract embeddings from dataset')
    parser.add_argument('--person', type=str, default=None,
                        help='Process only this person')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess all')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Model to use for embeddings')
    return parser.parse_args()


def calculate_folder_hash(folder_path):
    if not os.path.exists(folder_path):
        return None

    files = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            stat = os.stat(fpath)
            files.append(f"{fname}:{stat.st_size}:{stat.st_mtime}")

    if not files:
        return None

    combined = "|".join(files)
    return hashlib.md5(combined.encode()).hexdigest()


def needs_reprocessing(person_path, suffix):
    images_dir = os.path.join(person_path, "images")
    augmented_dir = os.path.join(person_path, "augmented")
    metadata_path = os.path.join(person_path, f"metadata_{suffix}.json")
    embeddings_path = os.path.join(person_path, f"embeddings_{suffix}.npz")

    if not os.path.exists(embeddings_path):
        return True, "missing embeddings"

    images_hash = calculate_folder_hash(images_dir)
    augmented_hash = calculate_folder_hash(augmented_dir)

    if not os.path.exists(metadata_path):
        return "update_metadata", {"images_hash": images_hash, "augmented_hash": augmented_hash}

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        saved_images_hash = metadata.get("images_hash")
        saved_augmented_hash = metadata.get("augmented_hash")
        saved_suffix = metadata.get("model_suffix")

        if not saved_images_hash or not saved_augmented_hash:
            return "update_metadata", {"images_hash": images_hash, "augmented_hash": augmented_hash}

        if saved_suffix != suffix:
            return True, f"model changed ({saved_suffix} -> {suffix})"

        if saved_images_hash != images_hash:
            return True, "images folder changed"

        if saved_augmented_hash != augmented_hash:
            return True, "augmented folder changed"

        return False, "up to date"

    except (json.JSONDecodeError, KeyError):
        return True, "corrupted metadata"


def process_dataset(dataset_dir: str,
                    detector: FaceDetector,
                    recognizer: FaceRecognizer,
                    model_name: str,
                    force_reprocess: bool = False,
                    specific_person: str = None):
    dataset_dir = os.path.abspath(dataset_dir)

    model_config = get_model_config(model_name)
    suffix = model_config['embeddings_suffix']

    people = os.listdir(dataset_dir)
    if specific_person:
        if specific_person not in people:
            print(f"Person '{specific_person}' not found in dataset")
            return
        people = [specific_person]

    processed = 0
    skipped = 0

    for person_name in people:
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        images_dir = os.path.join(person_path, "images")
        augmented_dir = os.path.join(person_path, "augmented")
        embeddings_path = os.path.join(person_path, f"embeddings_{suffix}.npz")
        metadata_path = os.path.join(person_path, f"metadata_{suffix}.json")

        if not os.path.exists(images_dir):
            print(f"SKIP {person_name}: missing images/")
            continue

        if not force_reprocess:
            check_result = needs_reprocessing(person_path, suffix)

            if check_result[0] == "update_metadata":
                print(f"UPDATE {person_name}: updating metadata")
                try:
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        n_orig = len([f for f in os.listdir(images_dir)
                                      if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]) if os.path.exists(images_dir) else 0
                        n_aug = len([f for f in os.listdir(augmented_dir)
                                     if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]) if os.path.exists(augmented_dir) else 0

                        metadata = {
                            "person": person_name,
                            "model_name": model_name,
                            "model_suffix": suffix,
                            "total_images": n_orig + n_aug,
                            "original_images": n_orig,
                            "augmented_images": n_aug,
                            "skipped_images": []
                        }

                    metadata.update(check_result[1])
                    metadata["model_suffix"] = suffix

                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    print(f"OK {person_name}: metadata updated")
                    skipped += 1
                    continue
                except Exception as e:
                    print(
                        f"WARN {person_name}: could not update metadata, reprocessing. Error: {e}")

            elif not check_result[0]:
                print(f"SKIP {person_name}: {check_result[1]}")
                skipped += 1
                continue
            else:
                print(f"PROCESS {person_name}: {check_result[1]}")
        else:
            print(f"FORCE {person_name}: force reprocess")

        image_files = []
        for folder, tag in [(images_dir, "original"), (augmented_dir, "augmented")]:
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    image_files.append((os.path.join(folder, fname), tag))

        if not image_files:
            print(f"WARN {person_name}: no images found")
            continue

        embeddings = {}
        metadata = {
            "person": person_name,
            "model_name": model_name,
            "model_suffix": suffix,
            "total_images": 0,
            "original_images": 0,
            "augmented_images": 0,
            "skipped_images": [],
            "images_hash": calculate_folder_hash(images_dir),
            "augmented_hash": calculate_folder_hash(augmented_dir)
        }

        for img_path, img_type in image_files:
            fname = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                metadata["skipped_images"].append(fname)
                continue

            faces = detector.detect(img)
            if len(faces) != 1:
                print(f"WARN {person_name}/{fname}: {len(faces)} faces, skip")
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
            print(f"OK {person_name}: {metadata['total_images']} embeddings "
                  f"({metadata['original_images']} orig, {metadata['augmented_images']} aug)")
            processed += 1
        else:
            print(f"WARN {person_name}: no embeddings saved")

    print(f"\n=== SUMMARY ===")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


def main():
    args = parse_args()

    dataset_dir = str(DATASET_DIR)
    detector_model_path = str(DETECTOR_MODEL_PATH)
    model_name = args.model

    print(f"Starting embedding extraction")
    print(f"Model: {model_name}")

    detector, recognizer = initialization_detector_recognizer(
        detector_model_path,
        model_name=model_name
    )

    process_dataset(
        dataset_dir,
        detector,
        recognizer,
        model_name,
        force_reprocess=args.force,
        specific_person=args.person
    )

    detector.close()
    recognizer.close()


if __name__ == "__main__":
    main()
