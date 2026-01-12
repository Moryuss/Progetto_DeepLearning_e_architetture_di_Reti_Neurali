from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
# pyright: ignore[reportMissingImports]
from facenet_pytorch import InceptionResnetV1


from facenet_pytorch import InceptionResnetV1
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer


def initialization_detector_recognizer(
    yolo_model_path,
    recognizer_model_path,
    backbone_type="InceptionResnetV1",
    backbone=None
):
    '''
    Inizializza e ritorna detector e recognizer

    :param yolo_model_path: Path da cui prendere i pesi del detector 
    :param recognizer_model_path: Path da cui prendere i pesi del recognizer
    :param backbone_type: Tipo di backbone da usare ("InceptionResnetV1", "custom", ecc.)
    :param backbone: Istanza custom del backbone (se None, viene creato in base a backbone_type)
    '''
    detector = FaceDetector(model_path=yolo_model_path)

    # Se viene passato un backbone custom, usalo direttamente
    if backbone is not None:
        recognizer = FaceRecognizer(
            model=backbone, model_path=recognizer_model_path)
    # Altrimenti crea il backbone in base al tipo
    elif backbone_type == "InceptionResnetV1":
        backbone = InceptionResnetV1(pretrained=None)
        recognizer = FaceRecognizer(
            model=backbone, model_path=recognizer_model_path)
    # Aggiungi qui altri tipi predefiniti se necessario
    # elif backbone_type == "MobileNetV2":
    #     backbone = MobileNetV2(...)
    #     recognizer = FaceRecognizer(model=backbone, model_path=recognizer_model_path)
    else:
        raise ValueError(f"Backbone type '{backbone_type}' non supportato")

    return detector, recognizer
