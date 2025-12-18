import torch
import numpy as np
import cv2
from src.utils import preprocess_for_recognizer
from src.config import (
    RECOGNIZER_MODEL_PATH
)


class FaceRecognizer:
    """
    Face recognizer basato su embedding vettoriali.
    Estrae un embedding normalizzato per ogni volto.
    """

    def __init__(
        self,
        model,
        model_path: str = str(RECOGNIZER_MODEL_PATH),
        device: str | None = None,
        image_size: int = 160
    ):
        """
        Args:
            model: istanza del modello PyTorch (architettura già creata)
            model_path: percorso al file .pt dei pesi
            device: 'cpu' o 'cuda' (auto se None)
            image_size: dimensione input del modello
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size

        # Sposta il modello sul device
        self.model = model.to(self.device)

        # Carica i pesi
        state_dict = torch.load(model_path, map_location=self.device)

        # Rimuovi eventuali pesi della testa di classificazione. Dato che voglio solo embedding
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("logits.")}

        self.model.load_state_dict(state_dict)

        # Modalità inference
        self.model.eval()

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocessing standard fatto falle utils
        """
        face_np = preprocess_for_recognizer(
            face_bgr,
            input_size=(self.image_size, self.image_size)
        )

        return torch.from_numpy(face_np).float().to(self.device)

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Estrae embedding normalizzato L2.

        Args:
            face_bgr: immagine del volto già ritagliata (H, W, 3)

        Returns:
            np.ndarray: embedding vettoriale (es. 512,)
        """
        face_tensor = self._preprocess(face_bgr)

        with torch.no_grad():

            embedding = self.model(face_tensor)

            embedding = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(embedding)

            embedding = embedding / norm if norm > 0 else embedding

            return embedding

    def close(self):
        """Libera il modello"""
        del self.model
        torch.cuda.empty_cache()
