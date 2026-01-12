import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import os
import torch  # pyright: ignore[reportMissingImports]
from src.config import (
    MODEL_NAME  # fallback della ricerca del nome del modello usato
)


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


def load_dataset_embeddings(dataset_dir: str, model_name: str = None, recognizer=None):
    """
    Carica tutti gli embeddings del dataset in memoria.

    Args:
        dataset_dir: path del dataset
        model_name: nome del modello (opzionale)
        recognizer: recognizer object per auto-rilevare il modello (opzionale)

    Ritorna:
        embeddings_list: lista di np.array
        labels_list: lista di nomi corrispondenti
    """
    # Determina il nome del modello
    if model_name is None:
        if recognizer is not None:
            model_name = get_model_name(recognizer)
        else:
            model_name = MODEL_NAME if MODEL_NAME is not None else "UnknownModel"

    embeddings_list = []
    labels_list = []

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)

        if not os.path.isdir(person_path):
            continue

        npz_path = os.path.join(person_path, f"embeddings_{model_name}.npz")

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            for fname, emb in data.items():
                embeddings_list.append(emb)
                labels_list.append(person_name)
        else:
            # Se non trova il file con model_name, prova con il vecchio nome
            old_npz_path = os.path.join(person_path, "embeddings.npz")
            if os.path.exists(old_npz_path):
                print(
                    f"[WARN] {person_name}: file con model_name non trovato, usando vecchio formato embeddings.npz")
                data = np.load(old_npz_path)
                for fname, emb in data.items():
                    embeddings_list.append(emb)
                    labels_list.append(person_name)
            else:
                print(f"[SKIP] {person_name}: nessun file embeddings trovato")

    if embeddings_list:
        embeddings_array = np.stack(embeddings_list)
    else:
        embeddings_array = np.array([])

    return embeddings_array, labels_list


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
