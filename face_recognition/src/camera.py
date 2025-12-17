import cv2


class Camera:
    def __init__(self, source=0, width=None, height=None):
        """
        source: int (webcam index) | str (video file / stream)
        width, height: opzionali, risoluzione desiderata
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la sorgente video: {source}")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """
        Ritorna:
            frame (np.ndarray) oppure None se fallisce
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Rilascia la risorsa video"""
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release()
