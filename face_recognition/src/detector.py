import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    def __init__(self, model_path: str, min_detection_confidence: float = 0.5):
        """
        FaceDetector basato su MediaPipe Tasks.

        Args:
            model_path: percorso al file .task
            min_detection_confidence: soglia minima di rilevamento [0,1]
        """
        # Base options per il modello
        base_options = python.BaseOptions(model_asset_path=model_path)

        # Configurazione detector
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )

        # Creazione del detector
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame):
        """
        Rileva volti in un frame BGR.

        Args:
            frame: np.ndarray (H, W, 3), immagine BGR

        Returns:
            List[Tuple[int, int, int, int]]: lista di bounding box (x1, y1, x2, y2)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Conversione in MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.detector.detect(mp_image)

        bboxes = []
        if results.detections:
            for detection in results.detections:
                box = detection.bounding_box
                x1 = int(box.origin_x)
                y1 = int(box.origin_y)
                x2 = int(box.origin_x + box.width)
                y2 = int(box.origin_y + box.height)
                bboxes.append((x1, y1, x2, y2))

        return bboxes

    def close(self):
        """Chiude il detector e libera risorse"""
        self.detector.close()

    def detect_with_keypoints(self, frame):
        """
        Rileva volti con keypoints.

        Returns:
            List[Dict]: lista di dizionari con 'bbox' e 'keypoints'
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        faces = []
        if results.detections:
            for detection in results.detections:
                # Bounding box
                box = detection.bounding_box
                bbox = (
                    int(box.origin_x),
                    int(box.origin_y),
                    int(box.origin_x + box.width),
                    int(box.origin_y + box.height)
                )

                # Keypoints (se disponibili)
                keypoints = []
                if detection.keypoints:
                    for kp in detection.keypoints:
                        keypoints.append({
                            'x': int(kp.x * frame.shape[1]),
                            'y': int(kp.y * frame.shape[0])
                        })

                faces.append({
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'score': detection.categories[0].score if detection.categories else 0.0
                })

        return faces
