# Puoi eseguire questo snippet velocemente?
import torch
# ""model_path": BASE_DIR / "models" / "face_recognition" / "vggface2.pt"
PERSONALIZED_RECOGNIZER_MODEL_PATH = "C:/Users/Matteo Ghidini/OneDrive - unibs.it/UniBs/MAGISTRALE/Deep Learning e Architetture di reti neurali/Code/face_recognition/models/face_recognition/best_model.pth"
# Sostituisci con il path di uno dei tuoi .pth
checkpoint = torch.load(
    PERSONALIZED_RECOGNIZER_MODEL_PATH, map_location="cpu")

print("Tipo:", type(checkpoint))
print("Keys:" if isinstance(checkpoint, dict) else "È un modello completo")
if isinstance(checkpoint, dict):
    print(list(checkpoint.keys())[:10])  # Prime 10 chiavi
