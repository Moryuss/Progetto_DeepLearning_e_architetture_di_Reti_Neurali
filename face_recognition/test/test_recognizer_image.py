import cv2
import numpy as np
import os

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from facenet_pytorch import InceptionResnetV1


def main():
    image_path = "data/test_image/matteo_test_image.jpg"

    if not os.path.exists(image_path):
        print(f"File non trovato: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Errore nel caricamento dell'immagine")
        return

    # === Init detector ===
    detector = FaceDetector(
        model_path="models/face_detection/yolo11_nano.pt",
        min_detection_confidence=0.5
    )

    # === Init recognizer ===
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone,
        model_path="models/face_recognition/vggface2.pt"
    )

    faces = detector.detect(image)

    for face in faces:
        print(face)
        (x1, y1, x2, y2) = face['bbox']
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        embedding = recognizer.get_embedding(face_crop)
        # per debug: deve essere circa 1
        norm = np.linalg.norm(embedding)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"emb_norm={norm:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("Image Face Recognition Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detector.close()
    recognizer.close()


if __name__ == "__main__":
    main()
