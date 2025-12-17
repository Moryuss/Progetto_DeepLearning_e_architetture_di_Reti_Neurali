import cv2
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path: str = 'yolov8n-face.pt', min_detection_confidence: float = 0.5):
        """
        FaceDetector basato su YOLO (Ultralytics).

        Args:
            model_path: percorso al file .pt (es. 'yolov8n-face.pt' o 'yolov11n-face.pt')
            min_detection_confidence: soglia minima di confidenza [0,1]
        """
        # Carica il modello YOLO
        self.model = YOLO(model_path)
        self.conf = min_detection_confidence

    def detect(self, frame):
        """
        Rileva volti in un frame BGR.

        Returns:
            dictionary con:
            List[Tuple[int, int, int, int]]: lista di bounding box (x1, y1, x2, y2)
            score: confidenza della rilevazione
        """
        # Esegue l'inferenza
        results = self.model(frame, conf=self.conf, verbose=False)
        faces = []

        for result in results:
            if not result.boxes:
                continue

            # Estraiamo i dati principali
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])

                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'score': float(scores[i])
                })

        # ritrona unico dizionario con bboxes e scores
        return faces

    def detect_with_keypoints(self, frame):
        """
        Rileva volti con keypoints usando YOLO11-Face.
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        faces = []

        for result in results:
            if not result.boxes:
                continue

            # Estraiamo i dati principali
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            # Recupero dei Keypoints (Landmarks)
            # In YOLO11-face sono tipicamente 5 punti per volto
            if result.keypoints is not None:
                # .xy restituisce le coordinate pixel (N_volti, N_punti, 2)
                kpts_all = result.keypoints.xy.cpu().numpy()
            else:
                kpts_all = [[] for _ in range(len(boxes))]

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])

                # Gestione sicura dei keypoints per ogni singola faccia
                formatted_kpts = []
                if i < len(kpts_all):
                    for kp in kpts_all[i]:
                        # Ignora i punti (0,0) che indicano landmark non rilevati
                        if kp[0] > 0 or kp[1] > 0:
                            formatted_kpts.append({
                                'x': int(kp[0]),
                                'y': int(kp[1])
                            })

                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'keypoints': formatted_kpts,
                    'score': float(scores[i])
                })

        return faces

    def close(self):
        """Metodo per compatibilità, YOLO gestisce la memoria automaticamente"""
        pass
