# Puoi eseguire questo snippet velocemente?
import torch
# ""model_path": BASE_DIR / "models" / "face_recognition" / "vggface2.pt"
PERSONALIZED_RECOGNIZER_MODEL_PATH = "C:/Users/Matteo Ghidini/OneDrive - unibs.it/UniBs/MAGISTRALE/Deep Learning e Architetture di reti neurali/Code/face_recognition/models/face_recognition/cnn_transferLearning_finetuned_celebA.pth"
# PERSONALIZED_RECOGNIZER_MODEL_PATH = "C:/Users/Matteo Ghidini/OneDrive - unibs.it/UniBs/MAGISTRALE/Deep Learning e Architetture di reti neurali/Code/face_recognition/models/face_recognition/dnn_lfw_(256).pth"

checkpoint = torch.load(
    PERSONALIZED_RECOGNIZER_MODEL_PATH, map_location="cpu")

print("Tipo:", type(checkpoint))
print("Keys:" if isinstance(checkpoint, dict) else "È un modello completo")
if isinstance(checkpoint, dict):
    print(list(checkpoint.keys())[:10])  # Prime 10 chiavi


checkpoint = torch.load(PERSONALIZED_RECOGNIZER_MODEL_PATH, map_location='cpu')

print("=== CONFIG COMPLETA ===")
print(checkpoint['config'])

print("\n=== CAMPI SPECIFICI ===")
config = checkpoint['config']
print(f"num_filters: {config.get('num_filters', 'MANCA!')}")
print(f"embedding_size: {config.get('embedding_size', 'MANCA!')}")
print(f"use_global_avg_pool: {config.get('use_global_avg_pool', 'MANCA!')}")
print(f"use_residual: {config.get('use_residual', 'MANCA!')}")

print("\n=== VERIFICA SHAPE PESI ===")
state_dict = checkpoint['model_state_dict']
print(f"conv_blocks.0.weight: {state_dict['conv_blocks.0.weight'].shape}")
print(f"conv_blocks.5.weight: {state_dict['conv_blocks.5.weight'].shape}")
print(f"conv_blocks.10.weight: {state_dict['conv_blocks.10.weight'].shape}")
