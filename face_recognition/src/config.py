from pathlib import Path

# root del progetto, face_recognition/
BASE_DIR = Path(__file__).resolve().parent.parent

# dataset
DATASET_DIR = BASE_DIR / "data" / "dataset"
CLASSIFY_IMAGES_DIR = BASE_DIR / "data" / "classify_images"

# modelli
DETECTOR_MODEL_PATH = BASE_DIR / "models" / "face_detection" / "yolo11_nano.pt"
RECOGNIZER_MODEL_PATH = BASE_DIR / "models" / "face_recognition" / "vggface2.pt"

# output
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
