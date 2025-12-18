import os
import cv2
import random
import numpy as np
from pathlib import Path


# =========================
# CONFIG
# =========================

DATASET_DIR = "data/dataset"
AUGMENT_THRESHOLD = 20    # numero minimo di immagini augmentate per persona
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================
# AUGMENTATION OPS
# =========================

def random_flip(img):
    if random.random() < 0.5:
        return cv2.flip(img, 1)
    return img


def random_brightness(img):
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def random_crop(img, crop_ratio=0.9):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y = random.randint(0, h - ch)
    x = random.randint(0, w - cw)
    cropped = img[y:y+ch, x:x+cw]
    return cv2.resize(cropped, (w, h))


def augment_image(img):
    img = random_flip(img)
    img = random_brightness(img)
    img = random_crop(img)
    return img


# =========================
# CORE LOGIC
# =========================

def list_images(folder: Path):
    return [
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def augment_person(person_dir: Path, threshold: int):
    images_dir = person_dir / "images"
    augmented_dir = person_dir / "augmented"

    if not images_dir.exists():
        print(f"[SKIP] {person_dir.name}: manca images/")
        return

    augmented_dir.mkdir(exist_ok=True)

    originals = list_images(images_dir)
    augmented = list_images(augmented_dir)

    if len(originals) == 0:
        print(f"[SKIP] {person_dir.name}: nessuna immagine originale")
        return

    if len(augmented) >= threshold:
        print(f"[OK] {person_dir.name}: già {len(augmented)} augmentate")
        return

    to_generate = threshold - len(augmented)
    print(f"[INFO] {person_dir.name}: genero {to_generate} immagini")

    aug_index = len(augmented)

    for i in range(to_generate):
        # sceglie random tra le imamgini originali
        src_path = random.choice(originals)
        img = cv2.imread(str(src_path))
        if img is None:
            continue

        aug_img = augment_image(img)

        aug_index += 1
        out_name = f"{src_path.stem}_aug_{aug_index:03d}.jpg"
        out_path = augmented_dir / out_name

        cv2.imwrite(str(out_path), aug_img)


def augment_dataset(dataset_dir: str, threshold: int):
    dataset_path = Path(dataset_dir)

    for person_dir in dataset_path.iterdir():
        if not person_dir.is_dir():
            continue
        augment_person(person_dir, threshold)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    augment_dataset(DATASET_DIR, AUGMENT_THRESHOLD)
