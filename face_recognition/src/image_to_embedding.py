import os
import json
import hashlib
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
    MODEL_NAME
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def calculate_folder_hash(folder_path):
    """Calcola hash MD5 dei file in una cartella per rilevare cambiamenti"""
    if not os.path.exists(folder_path):
        return None

    files = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            # Hash: nome file + size + mtime
            stat = os.stat(fpath)
            files.append(f"{fname}:{stat.st_size}:{stat.st_mtime}")

    if not files:
        return None

    combined = "|".join(files)
    return hashlib.md5(combined.encode()).hexdigest()


def needs_reprocessing(person_path, model_name):
    """Verifica se una persona necessita di riprocessing degli embeddings"""
    images_dir = os.path.join(person_path, "images")
    augmented_dir = os.path.join(person_path, "augmented")
    metadata_path = os.path.join(person_path, f"metadata_{model_name}.json")
    embeddings_path = os.path.join(person_path, f"embeddings_{model_name}.npz")

    # Se non esistono embedding, riprocessa
    if not os.path.exists(embeddings_path):
        return True, "missing embeddings"

    # Calcola hash correnti
    images_hash = calculate_folder_hash(images_dir)
    augmented_hash = calculate_folder_hash(augmented_dir)

    # Se metadata non esiste ma embedding sì, AGGIORNA metadata senza riprocessare, utile se sbaglio qualcosa o per la prima grande migrazione
    if not os.path.exists(metadata_path):
        return "update_metadata", {"images_hash": images_hash, "augmented_hash": augmented_hash}

    # Leggi hash salvati
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        saved_images_hash = metadata.get("images_hash")
        saved_augmented_hash = metadata.get("augmented_hash")
        saved_model = metadata.get("model_name")

        # Se mancano hash nel metadata, AGGIORNA
        if not saved_images_hash or not saved_augmented_hash:
            return "update_metadata", {"images_hash": images_hash, "augmented_hash": augmented_hash}

        # Verifica se modello è cambiato
        if saved_model != model_name:
            return True, f"model changed ({saved_model} -> {model_name})"

        # Verifica se immagini sono cambiate
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
    """
    Processa il dataset in modo incrementale

    Args:
        dataset_dir: directory del dataset
        detector: face detector
        recognizer: face recognizer
        model_name: nome del modello
        force_reprocess: forza il riprocessing di tutti
        specific_person: processa solo questa persona (None = tutte)
    """
    dataset_dir = os.path.abspath(dataset_dir)

    people = os.listdir(dataset_dir)
    if specific_person:
        if specific_person not in people:
            print(f"[ERROR] Person '{specific_person}' not found in dataset")
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
        embeddings_path = os.path.join(
            person_path, f"embeddings_{model_name}.npz")
        metadata_path = os.path.join(
            person_path, f"metadata_{model_name}.json")

        if not os.path.exists(images_dir):
            print(f"[SKIP] {person_name}: manca images/")
            continue

        # Verifica se serve riprocessare
        if not force_reprocess:
            check_result = needs_reprocessing(person_path, model_name)

            # Se serve solo aggiornare metadata (embedding già ok)
            if check_result[0] == "update_metadata":
                print(f"[UPDATE] {person_name}: updating metadata with hashes")
                try:
                    # Carica metadata esistente
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        # Conta immagini per creare metadata base
                        n_orig = len([f for f in os.listdir(images_dir)
                                      if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]) if os.path.exists(images_dir) else 0
                        n_aug = len([f for f in os.listdir(augmented_dir)
                                     if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]) if os.path.exists(augmented_dir) else 0

                        metadata = {
                            "person": person_name,
                            "model_name": model_name,
                            "total_images": n_orig + n_aug,
                            "original_images": n_orig,
                            "augmented_images": n_aug,
                            "skipped_images": []
                        }

                    # Aggiungi hash
                    metadata.update(check_result[1])

                    # Salva
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    print(f"[OK] {person_name}: metadata updated")
                    skipped += 1
                    continue
                except Exception as e:
                    print(
                        f"[WARN] {person_name}: could not update metadata, reprocessing. Error: {e}")
                    # Fallback: riprocessa

            elif not check_result[0]:  # False = up to date
                print(f"[SKIP] {person_name}: {check_result[1]}")
                skipped += 1
                continue
            else:  # True = needs reprocessing
                print(f"[PROCESS] {person_name}: {check_result[1]}")
        else:
            print(f"[FORCE] {person_name}: force reprocess")

        # Raccoglie immagini
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
            processed += 1
        else:
            print(f"[WARN] {person_name}: nessun embedding salvato")

    print(f"\n=== SUMMARY ===")
    print(f"Processed: {processed}")
    print(f"Skipped (up-to-date): {skipped}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract embeddings from dataset')
    parser.add_argument('--person', type=str, help='Process only this person')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess all')
    args = parser.parse_args()

    dataset_dir = str(DATASET_DIR)
    detector_model_path = str(DETECTOR_MODEL_PATH)
    recognizer_model_path = str(RECOGNIZER_MODEL_PATH)

    detector, recognizer = initialization_detector_recognizer(
        detector_model_path, recognizer_model_path)

    model_name = get_model_name(recognizer)
    print(f"[INFO] Using model: {model_name}")

    process_dataset(
        str(dataset_dir),
        detector,
        recognizer,
        model_name,
        force_reprocess=args.force,
        specific_person=args.person
    )

    detector.close()


if __name__ == "__main__":
    main()
