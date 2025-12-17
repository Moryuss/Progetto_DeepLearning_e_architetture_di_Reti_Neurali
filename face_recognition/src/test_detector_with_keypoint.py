import cv2
# Assicurati che in detector.py ci sia la versione YOLO
from detector import FaceDetector
from camera import Camera

# Percorso al modello YOLO (.pt)
model_path = "models/yolo11_nano.pt"
# model_path = "models/yolo8_medium.pt"

# Inizializza detector
detector = FaceDetector(model_path=model_path, min_detection_confidence=0.5)

# Inizializza la webcam
cam = Camera(0)

print(f"Usando modello: {model_path}")
print("Premi 'q' per uscire")

try:
    while True:
        frame = cam.read()
        if frame is None:
            break

        # Il metodo .detect_with_keypoints() restituisce dizionario con (x1, y1, x2, y2), keypoints e score
        faces = detector.detect_with_keypoints(frame)
        # DEBUG
        print(f"Volti rilevati: {len(faces)}")
        print(faces)

        for face in faces:
            # Dati volto
            (x1, y1, x2, y2) = face['bbox']
            keypoints = face['keypoints']
            score = face['score']

            # Bounding box e score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Keypoints
            for kp in keypoints:
                # kp è un dizionario: {'x': val, 'y': val}
                cv2.circle(frame, (kp['x'], kp['y']), 3, (0, 0, 255), -1)

        # Mostra il frame
        cv2.imshow("YOLO Face Detection", frame)

        # Esci con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.release()
    detector.close()
    cv2.destroyAllWindows()
