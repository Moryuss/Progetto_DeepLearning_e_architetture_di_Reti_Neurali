import cv2
from detector import FaceDetector
import os


def main():
    # DEBUG: stampa cartella corrente
    print("Current working directory:", os.getcwd())
    image_path = "data/test_image/matteo_test_image.jpg"

    # Carica l'immagine
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Errore: impossibile aprire {image_path}")
        return

    # Crea il rilevatore di volti
    detector = FaceDetector(
        model_path="models/face_detection/yolo11_nano.pt", min_detection_confidence=0.5)

    # Rileva i volti
    faces = detector.detect(frame)

    # Disegna bounding box e punteggi
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        score = face['score']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostra l'immagine
    cv2.imshow("Faces", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Chiudi il detector
    detector.close()


if __name__ == "__main__":
    main()
