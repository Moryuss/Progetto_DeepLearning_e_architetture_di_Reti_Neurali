from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
# pyright: ignore[reportMissingImports]
from facenet_pytorch import InceptionResnetV1


def initialization_detector_recognizer(yolo_model_path, recognizer_model_path):
    '''
    Inizializza e ritorna detector e recognizer    
    :param yolo_model_path: Path da cui prendere i pesi del detector 
    :param recognizer_model_path: Path da cui prendere i pesi del recognizer
    '''
    detector = FaceDetector(model_path=yolo_model_path)
    # basta modificare questo per cambaire il Modello. Qui dovrò mettere il mio modello esportato e anche i pesi salvarli in model
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone, model_path=recognizer_model_path)

    return detector, recognizer
