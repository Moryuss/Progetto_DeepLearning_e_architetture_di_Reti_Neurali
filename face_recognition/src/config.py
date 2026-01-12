from pathlib import Path

# root del progetto, face_recognition/
BASE_DIR = Path(__file__).resolve().parent.parent

# dataset
DATASET_DIR = BASE_DIR / "data" / "dataset"
CLASSIFY_IMAGES_DIR = BASE_DIR / "data" / "classify_images"

# modelli
MODEL_NAME = "ResNet50"
DETECTOR_MODEL_PATH = BASE_DIR / "models" / "face_detection" / "yolo11_nano.pt"
RECOGNIZER_MODEL_PATH = BASE_DIR / "models" / "face_recognition" / "vggface2.pt"

# output
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"


# similarity
PEOPLE_EMB_PATH = BASE_DIR / "data" / \
    "similarity_images" / "embeddings" / "people.npz"
KNOWN_EMB_PATH = BASE_DIR / "data" / \
    "similarity_images" / "embeddings" / "known.npz"

PEOPLE_DIR = BASE_DIR / "data" / "similarity_images" / "people"
KNOWN_PEOPLE_DIR = BASE_DIR / "data" / "similarity_images" / "known_people"
