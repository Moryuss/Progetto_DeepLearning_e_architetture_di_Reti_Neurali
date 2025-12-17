import cv2
# Assicurati che in detector.py ci sia la versione YOLO
from detector import FaceDetector
from camera import Camera

# Percorso al modello YOLO (.pt invece di .tflite)
# Se non hai il file, Ultralytics lo scaricherà automaticamente al primo avvio
model_path = "models/yolo8_nano.pt"


# Inizializza detector (l'interfaccia rimane identica)
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

        # Il metodo .detect() restituisce (x1, y1, x2, y2) proprio come prima
        bboxes = detector.detect(frame)

        # Disegna bounding box
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Opzionale: aggiungi una label
            cv2.putText(frame, "Face", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostra il frame
        cv2.imshow("YOLO Face Detection", frame)

        # Esci con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.release()
    detector.close()
    cv2.destroyAllWindows()
