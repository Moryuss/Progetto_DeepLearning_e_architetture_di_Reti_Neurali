from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.models_recognition import create_model


def initialization_detector_recognizer(
    yolo_model_path,
    recognizer_model_path=None,
    backbone_type=None,
    model_config=None,
    model_name=None
):
    """
    Inizializza e ritorna detector e recognizer.

    Args:
        yolo_model_path: Path ai pesi del detector YOLO
        recognizer_model_path: Path ai pesi del recognizer (opzionale se usi model_name)
        backbone_type: Tipo di backbone (opzionale se usi model_name)
        model_config: Dict con parametri per FaceEmbeddingCNN (opzionale)
        model_name: Nome del modello da AVAILABLE_MODELS (es. "CNN Baseline (128-dim)")

    Returns:
        tuple: (detector, recognizer)

    Examples:
        # Metodo 1: Usa model_name dal config (RACCOMANDATO)
        detector, recognizer = initialization_detector_recognizer(
            yolo_model_path="models/yolov11n-face.pt",
            model_name="CNN Baseline (128-dim)"
        )

        # Metodo 2: Specifica manualmente (vecchio modo)
        detector, recognizer = initialization_detector_recognizer(
            yolo_model_path="models/yolov11n-face.pt",
            recognizer_model_path="models/vggface2.pt",
            backbone_type="InceptionResnetV1"
        )
    """
    from src.config import get_model_config, DEFAULT_MODEL

    # Se model_name è specificato, carica config dal file
    if model_name:
        config = get_model_config(model_name)
        recognizer_model_path = str(config["model_path"])
        backbone_type = config["backbone_type"]
        print(f" Caricamento modello: {model_name}")
        print(f"   Tipo: {backbone_type}")
        print(f"   Path: {recognizer_model_path}")

    # Fallback: usa default se niente è specificato
    if recognizer_model_path is None:
        print(f"  Nessun modello specificato, uso default: {DEFAULT_MODEL}")
        config = get_model_config(DEFAULT_MODEL)
        recognizer_model_path = str(config["model_path"])
        backbone_type = config["backbone_type"]

    if backbone_type is None:
        backbone_type = "InceptionResnetV1"  # default vecchio

    # Inizializza detector
    detector = FaceDetector(model_path=yolo_model_path)

    # Crea il modello usando la factory
    if model_config is None:
        # Carica config dal checkpoint (se disponibile)
        model_info = create_model(
            backbone_type=backbone_type,
            checkpoint_path=recognizer_model_path
        )
    else:
        # Usa config fornita
        model_info = create_model(
            backbone_type=backbone_type,
            checkpoint_path=recognizer_model_path,
            **model_config
        )

    # Crea il recognizer con tutti i parametri necessari
    recognizer = FaceRecognizer(
        model=model_info['model'],
        model_path=recognizer_model_path,
        image_size=model_info['image_size'],
        embedding_size=model_info['embedding_size'],
        use_get_embedding=model_info['use_get_embedding']
    )

    return detector, recognizer
