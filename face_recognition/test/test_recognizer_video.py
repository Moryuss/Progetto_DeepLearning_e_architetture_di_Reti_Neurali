import cv2
import numpy as np

from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from facenet_pytorch import InceptionResnetV1


def main():
    # === Init detector ===
    detector = FaceDetector(
        model_path="yolov8n-face.pt",
        min_detection_confidence=0.5
    )

    # === Init recognizer ===
    # architettura, modello vuoto a cui dopo dò i pesi
    backbone = InceptionResnetV1(pretrained=None)
    recognizer = FaceRecognizer(
        model=backbone,
        # pesi del modello con architettura specificata sopra
        model_path="models/face_recognition/vggface2.pt"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: webcam non disponibile")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        for (x1, y1, x2, y2) in faces:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            embedding = recognizer.get_embedding(face_crop)

            # Debug: norma embedding (deve essere ~1)
            norm = np.linalg.norm(embedding)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"emb_norm={norm:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.imshow("Camera Face Recognition Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    recognizer.close()


if __name__ == "__main__":
    main()
