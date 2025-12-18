import os
import cv2
import numpy as np
from src.utils import load_embeddings

from src.config import (
    PEOPLE_EMB_PATH,
    KNOWN_EMB_PATH,
    PEOPLE_DIR,
    KNOWN_PEOPLE_DIR
)

# ---------------- CONFIG ----------------
TOP_K = 3
SIM_THRESHOLD = 0.65
MAX_IMG_SIZE = 300
# ----------------------------------------


def find_top_k(query_emb, db_embs, db_names, k):
    sims = db_embs @ query_emb
    idxs = np.argsort(sims)[::-1][:k]

    return [
        (db_names[i], float(sims[i]))
        for i in idxs
    ]


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


def main():
    # semplicemente si controlla la cosine similarity tra immagini già embedded
    # le immagini sono embeddate con extract_embeddings, non image_to_embedding

    # ---- load embeddings ----
    people_embs, people_names = load_embeddings(PEOPLE_EMB_PATH)
    known_embs, known_names = load_embeddings(KNOWN_EMB_PATH)

    print(f"[INFO] PEOPLE embeddings: {len(people_names)}")
    print(f"[INFO] KNOWN embeddings:  {len(known_names)}")

    # ---- iterate queries ----
    for q_name, q_emb in zip(known_names, known_embs):
        print(f"\n=== QUERY: {q_name} ===")

        top_matches = find_top_k(
            q_emb,
            people_embs,
            people_names,
            TOP_K
        )

        # ---- console output ----
        for rank, (name, sim) in enumerate(top_matches, 1):
            tag = "FOUND MATCH" if sim >= SIM_THRESHOLD else "NO MATCH"
            print(f"{tag} {rank}. {name:<30} sim={sim:.3f}")

        # ---- visualization ----
        q_img = load_image_safe(
            os.path.join(KNOWN_PEOPLE_DIR, q_name),
            MAX_IMG_SIZE
        )

        if q_img is None:
            continue

        rows = [q_img]
        for name, sim in top_matches:
            img = load_image_safe(
                os.path.join(PEOPLE_DIR, name),
                MAX_IMG_SIZE
            )
            if img is None:
                continue

            cv2.putText(
                img,
                f"{sim:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            rows.append(img)

        collage = cv2.hconcat(rows)
        cv2.imshow("Similarity Search", collage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
