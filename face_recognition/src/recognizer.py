import torch
import numpy as np
from src.utils import preprocess_for_recognizer


class FaceRecognizer:
    """
    Face recognizer basato su embedding vettoriali.
    Supporta diversi backbone (InceptionResnetV1, FaceEmbeddingCNN, custom).
    """

    def __init__(
        self,
        model,
        model_path: str,
        device: str | None = None,
        image_size: int = 160,
        embedding_size: int = 512,
        use_get_embedding: bool = False
    ):
        """
        Args:
            model: istanza del modello PyTorch (architettura già creata)
            model_path: percorso al file .pt/.pth dei pesi
            device: 'cpu' o 'cuda' (auto se None)
            image_size: dimensione input del modello (es. 160 per InceptionResnetV1, 128 per FaceEmbeddingCNN/DNN)
            embedding_size: dimensione output embedding (es. 512, 128, 256, etc.)
            use_get_embedding: se True, usa model.get_embedding(), altrimenti model.forward()
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size
        self.embedding_size = embedding_size
        self.use_get_embedding = use_get_embedding

        # Sposta il modello sul device
        self.model = model.to(self.device)

        # Carica i pesi
        checkpoint = torch.load(model_path, map_location=self.device)

        # Gestisci diversi formati di checkpoint
        if isinstance(checkpoint, dict):
            # Checkpoint con dizionario (model_state_dict, config, etc.)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume che sia direttamente lo state_dict
                state_dict = checkpoint
        else:
            # Checkpoint è direttamente il modello (raro ma possibile)
            state_dict = checkpoint.state_dict() if hasattr(
                checkpoint, 'state_dict') else checkpoint

        # Rimuovi eventuali pesi della testa di classificazione
        # (per modelli addestrati con classification head)
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("logits.") and not k.startswith("classifier.")
        }

        # Carica i pesi nel modello (strict=False per ignorare layer mancanti come classifier)
        self.model.load_state_dict(state_dict, strict=False)

        # Modalità inference
        self.model.eval()

        print(f" FaceRecognizer inizializzato:")
        print(f"  - Device: {self.device}")
        print(f"  - Image size: {self.image_size}x{self.image_size}")
        print(f"  - Embedding size: {self.embedding_size}")
        print(
            f"  - Metodo: {'get_embedding()' if use_get_embedding else 'forward()'}")

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocessing standard usando le utils.
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
            face_bgr: immagine del volto già ritagliata (H, W, 3) in formato BGR

        Returns:
            np.ndarray: embedding vettoriale normalizzato (es. shape: (512,) o (128,))
        """
        face_tensor = self._preprocess(face_bgr)

        with torch.no_grad():
            # Usa il metodo appropriato in base al modello
            if self.use_get_embedding:
                embedding = self.model.get_embedding(face_tensor)
            else:
                embedding = self.model(face_tensor)

            # Converti in numpy
            embedding = embedding.cpu().numpy()[0]

            # Normalizzazione L2
            norm = np.linalg.norm(embedding)
            embedding = embedding / norm if norm > 0 else embedding

            return embedding

    def close(self):
        """Libera il modello dalla memoria"""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
