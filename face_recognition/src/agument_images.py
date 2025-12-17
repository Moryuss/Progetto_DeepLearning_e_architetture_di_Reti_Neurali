import os
import cv2
import numpy as np


def augment_face(face_bgr):
    """
    Restituisce una lista di immagini augmentate a partire da face_bgr
    """
    augmented = []

    # Horizontal flip
    flipped = cv2.flip(face_bgr, 1)
    augmented.append(flipped)

    # Random crop ±10% dell'altezza/larghezza
    h, w, _ = face_bgr.shape
    dx = int(0.2 * w)
    dy = int(0.2 * h)
    x1 = np.random.randint(0, dx+1)
    y1 = np.random.randint(0, dy+1)
    x2 = w - np.random.randint(0, dx+1)
    y2 = h - np.random.randint(0, dy+1)
    crop = face_bgr[y1:y2, x1:x2]
    crop = cv2.resize(crop, (w, h))
    augmented.append(crop)

    # Brightness adjustment ±20%
    for factor in [0.6, 1.6]:
        bright = np.clip(face_bgr * factor, 0, 255).astype(np.uint8)
        augmented.append(bright)

    return augmented


def augment_dataset(dataset_dir: str):
    """
    Scansiona tutte le cartelle 'images' nel dataset e aggiunge versioni augmentate
    """
    dataset_dir = os.path.abspath(dataset_dir)

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        images_path = os.path.join(person_path, "images")
        if not os.path.isdir(images_path):
            continue

        for fname in os.listdir(images_path):
            img_path = os.path.join(images_path, fname)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Impossibile leggere {img_path}")
                continue

            augmented_images = augment_face(img)
            for idx, aug_img in enumerate(augmented_images):
                base, ext = os.path.splitext(fname)
                aug_fname = f"{base}_augmented{idx}{ext}"
                aug_path = os.path.join(images_path, aug_fname)
                cv2.imwrite(aug_path, aug_img)
                print(f"[INFO] Salvata immagine augmentata: {aug_path}")


def main():
    dataset_dir = "data/dataset"  # percorso al tuo dataset
    augment_dataset(dataset_dir)


if __name__ == "__main__":
    main()
