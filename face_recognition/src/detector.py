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
            List[Tuple[int, int, int, int]]: lista di bounding box (x1, y1, x2, y2)
        """
        # Esegue l'inferenza
        # stream=True è più efficiente per i video
        results = self.model(frame, conf=self.conf, verbose=False)

        bboxes = []
        for result in results:
            for box in result.boxes:
                # Coordinate come float e poi convertite in int
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bboxes.append((x1, y1, x2, y2))

        return bboxes

    def detect_with_keypoints(self, frame):
        """
        Rileva volti con keypoints (occhi, naso, orecchie, bocca).

        Returns:
            List[Dict]: lista di dizionari con 'bbox', 'keypoints' e 'score'
        """
        results = self.model(frame, conf=self.conf, verbose=False)

        faces = []
        for result in results:
            # result.boxes contiene le bbox, result.keypoints i punti del volto
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            # YOLO-face restituisce solitamente 5 o 6 keypoints
            if result.keypoints is not None:
                kpts_all = result.keypoints.xy.cpu().numpy()

                for bbox, score, kpts in zip(boxes, scores, kpts_all):
                    x1, y1, x2, y2 = map(int, bbox)

                    # Formatta i keypoints come nel tuo codice originale
                    formatted_kpts = []
                    for kp in kpts:
                        formatted_kpts.append({
                            'x': int(kp[0]),
                            'y': int(kp[1])
                        })

                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'keypoints': formatted_kpts,
                        'score': float(score)
                    })

        return faces

    def close(self):
        """Metodo per compatibilità, YOLO gestisce la memoria automaticamente"""
        pass
