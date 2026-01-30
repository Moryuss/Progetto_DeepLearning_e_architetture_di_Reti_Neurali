from pathlib import Path

# root del progetto, face_recognition/
BASE_DIR = Path(__file__).resolve().parent.parent

# dataset
DATASET_DIR = BASE_DIR / "data" / "dataset"
CLASSIFY_IMAGES_DIR = BASE_DIR / "data" / "classify_images"

TEST_IMAGES_DIR = BASE_DIR / "data" / "test_images"
# modelli detector (YOLO - non cambia)
DETECTOR_MODEL_PATH = BASE_DIR / "models" / "face_detection" / "yolo11_nano.pt"

# ============================================================================
# MODELLI DI RICONOSCIMENTO DISPONIBILI
# ============================================================================
AVAILABLE_MODELS = {
    "InceptionResnetV1 (VGGFace2)": {
        "backbone_type": "InceptionResnetV1",
        "model_path": BASE_DIR / "models" / "face_recognition" / "vggface2.pt",
        "embeddings_suffix": "InceptionResnetV1",  # suffisso per file embeddings
        "description": "Modello pretrained su VGGFace2 (512-dim embeddings)"
    },
    "CNN Baseline (LFW)(128-dim)": {
        "backbone_type": "FaceEmbeddingCNN",
        "model_path": BASE_DIR / "models" / "face_recognition" / "cnn_baseline_lfw.pth",
        "embeddings_suffix": "cnn_baseline_128",
        "description": "CNN custom con embeddings 128-dim LFW dataset"
    },
    "CNN Optimized (LFW)(256-dim)": {
        "backbone_type": "FaceEmbeddingCNN",
        "model_path": BASE_DIR / "models" / "face_recognition" / "cnn_baseline_LFW.pth",
        "embeddings_suffix": "cnn_optimized_256",
        "description": "CNN custom con embeddings 256-dim LFW dataset"
    },
    "CNN GAP (CelebA)(128-dim)": {
        "backbone_type": "FaceEmbeddingCNN",
        "model_path": BASE_DIR / "models" / "face_recognition" / "cnn_gap_celeba.pth",
        "embeddings_suffix": "cnn_gap_128",
        "description": "CNN custom con embeddings 128-dim e Global Avarage Pooling"
    },
    "CNN Deep Residual (CelebA) (256-dim)": {
        "backbone_type": "FaceEmbeddingCNN",
        "model_path": BASE_DIR / "models" / "face_recognition" / "cnn_deep_residual_celeba.pth",
        "embeddings_suffix": "cnn_deep_residual_256",
        "description": "CNN con residual blocks, embeddings 256-dim"
    },
    "CNN Transfer Learning & Fine Tuning (CelebA) (256-dim)": {
        "backbone_type": "TransferLearningEmbeddingCNN",
        "model_path": BASE_DIR / "models" / "face_recognition" / "cnn_transferLearning_finetuned_celebA.pth",
        "embeddings_suffix": "cnn_transferLearning_CelebA_256",
        "description": "CNN Resnet pretrained su ImageNet, transfer learning e fine-tuned su CelebA"
    },
    "CNN Transfer Learning & Fine Tuning (LFW) (256-dim)": {
        "backbone_type": "TransferLearningEmbeddingCNN",
        "model_path": BASE_DIR / "models" / "face_recognition" / "cnn_transferLearning_finetuned_LFW.pth",
        "embeddings_suffix": "cnn_transferLearning_LFW_256",
        "description": "CNN Resnet pretrained su ImageNet, transfer learning ee fine-tuned su LFW"
    },
}

# Modello di default (usato se non specificato)
DEFAULT_MODEL = "InceptionResnetV1 (VGGFace2)"

# ============================================================================
# BACKWARD COMPATIBILITY (vecchie variabili)
# ============================================================================
# Questi vengono usati se il codice vecchio non specifica il modello
MODEL_NAME = AVAILABLE_MODELS[DEFAULT_MODEL]["backbone_type"]
RECOGNIZER_MODEL_PATH = AVAILABLE_MODELS[DEFAULT_MODEL]["model_path"]

# output
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

# similarity
PEOPLE_EMB_PATH = BASE_DIR / "data" / \
    "similarity_images" / "embeddings" / "people.npz"
KNOWN_EMB_PATH = BASE_DIR / "data" / \
    "similarity_images" / "embeddings" / "known.npz"
PEOPLE_DIR = BASE_DIR / "data" / "similarity_images" / "people"
KNOWN_PEOPLE_DIR = BASE_DIR / "data" / "similarity_images" / "known_people"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_config(model_name):
    """
    Ottiene la configurazione di un modello dato il suo nome.

    Args:
        model_name: nome del modello dalla lista AVAILABLE_MODELS

    Returns:
        dict con backbone_type, model_path, embeddings_suffix, description
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Modello '{model_name}' non trovato. "
                         f"Modelli disponibili: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_name]


def get_embeddings_path(dataset_dir, model_name=None, person_name=None):
    """
    Genera il path per il file embeddings in base al modello.

    Args:
        dataset_dir: directory del dataset (es. data/dataset o data/dataset/alice)
        model_name: nome del modello (se None, usa DEFAULT_MODEL)
        person_name: nome della persona (opzionale, per dataset)

    Returns:
        Path al file .npz degli embeddings

    Examples:
        # Dataset globale
        get_embeddings_path(DATASET_DIR, "CNN Baseline (128-dim)")
        -> data/dataset/embeddings_cnn_baseline_128.npz

        # Persona specifica
        get_embeddings_path(DATASET_DIR / "alice", "CNN Baseline (128-dim)")
        -> data/dataset/alice/embeddings_cnn_baseline_128.npz
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    config = get_model_config(model_name)
    suffix = config["embeddings_suffix"]

    dataset_path = Path(dataset_dir)

    # Se è una persona specifica, metti embeddings nella sua cartella
    if person_name:
        return dataset_path / person_name / f"embeddings_{suffix}.npz"
    else:
        return dataset_path / f"embeddings_{suffix}.npz"


def get_people_embeddings_path(model_name=None):
    """
    Genera path per embeddings di people in base al modello.

    Returns:
        Path al file people_[suffix].npz
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    config = get_model_config(model_name)
    suffix = config["embeddings_suffix"]

    return PEOPLE_EMB_PATH.parent / f"people_{suffix}.npz"


def get_known_embeddings_path(model_name=None):
    """
    Genera path per embeddings di known_people in base al modello.

    Returns:
        Path al file known_{suffix}.npz
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    config = get_model_config(model_name)
    suffix = config["embeddings_suffix"]

    return KNOWN_EMB_PATH.parent / f"known_{suffix}.npz"
