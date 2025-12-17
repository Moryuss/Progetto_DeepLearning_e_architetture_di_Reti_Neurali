import cv2
from detector import FaceDetector
from camera import Camera

# Percorso al modello .task
model_path = "models/blaze_face_short_range.tflite"

# Inizializza detector
detector = FaceDetector(model_path=model_path, min_detection_confidence=0.5)

# Inizializza la webcam (0 = default)
cam = Camera(0)

print("Premi 'q' per uscire")

try:
    while True:
        frame = cam.read()
        if frame is None:
            break

        # Rileva volti
        bboxes = detector.detect(frame)

        # Disegna bounding box
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostra il frame
        cv2.imshow("Face Detector Tasks", frame)

        # Esci con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.release()
    detector.close()
    cv2.destroyAllWindows()
