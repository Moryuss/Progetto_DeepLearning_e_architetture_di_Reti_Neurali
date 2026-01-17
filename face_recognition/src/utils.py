import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import os
import torch  # pyright: ignore[reportMissingImports]
from src.config import (
    MODEL_NAME,  # fallback della ricerca del nome del modello usato
    DEFAULT_MODEL,
    get_model_config
)
from pathlib import Path


def crop_face(image, bbox, margin=0):
    """
    image: frame BGR (H, W, 3)
    bbox: (x1, y1, x2, y2)
    margin: pixel extra intorno al volto

    return: cropped face (BGR)
    """
    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return image[y1:y2, x1:x2]


def resize_image(image, size):
    """
    image: np.ndarray
    size: (width, height)

    return: resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def resize_max(image, max_dim=800):
    """
    Ridimensiona l'immagine in modo che la dimensione più grande sia max_dim
    Mantiene il rapporto d'aspetto
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))


def preprocess_for_yolo(image, input_size):
    """
    image: frame BGR
    input_size: (width, height)

    return: input tensor (1, 3, H, W)  (1 dimensione batch, 3 canali, altezza, larghezza)
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = resize_image(img, input_size)
    img = img.astype(np.float32) / 255.0
    # normalizza correttamente

    # HWC → CHW     /height, width, channels
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)   # → NCHW

    return img


def preprocess_for_recognizer(image, input_size):
    """
    image: face crop BGR
    input_size: (width, height)

    Preprocessing compatibile con:
    - FaceNet
    - ArcFace
    - InsightFace (PyTorch)

    return: input tensor (1, 3, H, W)
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = resize_image(img, input_size)
    img = img.astype(np.float32)

    # standard FaceNet / ArcFace --> trasforma da [0,255] a [-1,1]
    img = (img - 127.5) / 128.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def get_model_name(recognizer):
    """Estrae il nome del modello, con fallback a MODEL_NAME da config"""
    try:
        return recognizer.model.__class__.__name__
    except:
        return MODEL_NAME if MODEL_NAME is not None else "UnknownModel"


def load_dataset_embeddings(dataset_dir, recognizer=None, model_name=None):
    """
    Carica embeddings e label dal dataset.
    Usa il file embeddings corretto in base al modello.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
        print(f"No model specified, using default: {model_name}")

    dataset_path = Path(dataset_dir)

    all_embeddings = []
    all_labels = []

    # Ottieni suffix per il nuovo formato
    model_config = get_model_config(model_name)
    suffix = model_config['embeddings_suffix']

    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        # Prova NUOVO formato: embeddings_{suffix}.npz
        emb_file = person_dir / f"embeddings_{suffix}.npz"

        # FALLBACK: se non esiste, prova VECCHIO formato: embeddings.npz
        if not emb_file.exists():
            emb_file_old = person_dir / "embeddings.npz"
            if emb_file_old.exists():
                print(f"WARN {person_name}: using old format embeddings.npz")
                emb_file = emb_file_old
            else:
                print(f"SKIP {person_name}: no embeddings found")
                continue

        try:
            data = np.load(emb_file)

            # VECCHIO formato: dizionario {filename: embedding}
            if 'embeddings' not in data and 'labels' not in data:
                embeddings = np.array([data[key]
                                      for key in sorted(data.keys())])
                labels = [person_name] * len(embeddings)
            # NUOVO formato: array 'embeddings' e 'labels'
            else:
                embeddings = data['embeddings']
                labels = data['labels']

            all_embeddings.append(embeddings)
            all_labels.extend(labels)

            print(f"OK Loaded {len(embeddings)} embeddings for {person_name}")

        except Exception as e:
            print(f"ERROR loading embeddings for {person_name}: {e}")
            continue

    if not all_embeddings:
        print("ERROR No embeddings found! Run embedding extraction first.")
        return np.array([]), []

    embeddings_array = np.vstack(all_embeddings)

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / (norms + 1e-8)

    print(
        f"\nDataset loaded: {len(all_labels)} embeddings from {len(all_embeddings)} people")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")

    return embeddings_array, all_labels


def load_embeddings(npz_path, model_name: str = None):
    '''
    Carica embeddings da file .npz, non contiene nomi personali ma solo nomi file immagine.

    Args:
        npz_path: path base del file (senza suffisso modello)
        model_name: nome del modello (opzionale, usa MODEL_NAME se None)

    Ritorna: embeddings_array, names_list    
    '''
    if model_name is None:
        model_name = MODEL_NAME if MODEL_NAME is not None else "UnknownModel"

    # Converti Path a stringa se necessario
    npz_path = str(npz_path)

    # Prova prima con il nome del modello
    if not npz_path.endswith('.npz'):
        npz_path = f"{npz_path}.npz"

    # Costruisci path con model_name
    base_path = npz_path.replace('.npz', '')
    model_npz_path = f"{base_path}_{model_name}.npz"

    # Prova prima con il nuovo formato
    if os.path.exists(model_npz_path):
        data = np.load(model_npz_path)
    elif os.path.exists(npz_path):
        # Fallback al vecchio formato
        print(f"[WARN] Usando vecchio formato: {npz_path}")
        data = np.load(npz_path)
    else:
        raise FileNotFoundError(
            f"Nessun file trovato: {model_npz_path} o {npz_path}")

    names = list(data.keys())
    embs = np.vstack([data[k] for k in names])
    return embs, names


def recognize_faces(frame, detector, recognizer, embeddings_array, labels_list, threshold=0.60):
    """
    Rileva volti, calcola embedding, confronta con dataset.
    Restituisce lista di dict con bbox, nome e confidence
    """
    faces = detector.detect(frame)
    results = []

    for face in faces:
        (x1, y1, x2, y2) = face['bbox']
        face_crop = frame[y1:y2, x1:x2]
        name = "Unknown"
        confidence = 0.0

        emb = recognizer.get_embedding(face_crop)

        if embeddings_array.size > 0:
            # cosine similarity = dot product (embedding L2-normalizzati)
            sims = embeddings_array @ emb  # prodotto matriciale
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]

            if best_sim > threshold:
                name = labels_list[best_idx]
                # confidenza normalizzata in [0, 1]
                confidence = (best_sim - threshold) / (1.0 - threshold)

        results.append({
            "bbox": (x1, y1, x2, y2),
            "name": name,
            "confidence": confidence
        })

    return results


def draw_label(frame, name, confidence, bbox, color=(0, 255, 0), font_scale=2, thickness=2):
    """
    Disegna un rettangolo e il nome con confidenza sopra il volto.
    name: stringa del nome
    confidence: float [0,1]
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Testo
    label_text = f"{name} ({confidence:.2f})"

    # Rettangolo del volto
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Calcola dimensione testo
    text_size, _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    text_w, text_h = text_size

    # Posizione testo sopra il volto
    text_x = x1
    text_y = max(y1 - 5, text_h + 5)

    # Sfondo opaco per leggibilità
    cv2.rectangle(frame, (text_x, text_y - text_h - 2),
                  (text_x + text_w, text_y + 2), (0, 0, 0), cv2.FILLED)

    # Scrivi testo
    cv2.putText(frame, label_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# similarity methods


def load_image_safe(path, max_size=None):
    img = cv2.imread(path)
    if img is None:
        return None

    if max_size:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img

# per la rappresentazione con cv2


def resize_height(img, target_height):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))


def find_top_k(query_emb, db_embs, db_names, k):
    sims = db_embs @ query_emb
    idxs = np.argsort(sims)[::-1][:k]
    return [
        (db_names[i], float(sims[i]))
        for i in idxs
    ]
##
