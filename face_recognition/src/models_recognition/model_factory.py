import torch
from facenet_pytorch import InceptionResnetV1
from .face_embedding_cnn import FaceEmbeddingCNN
from .face_embedding_dnn import FaceEmbeddingDNN


def create_cnn_model(config):
    """
    Factory function per creare CNN con diverse configurazioni

    Args:
        config: dict con parametri del modello

    Returns:
        model: istanza di FaceEmbeddingCNN
    """
    return FaceEmbeddingCNN(
        input_channels=config.get('input_channels', 3),
        num_filters=config.get('num_filters', [32, 64, 128]),
        kernel_sizes=config.get('kernel_sizes', [3, 3, 3]),
        fc_hidden_size=config.get('fc_hidden_size', 512),
        embedding_size=config.get('embedding_size', 128),
        num_classes=config.get('num_classes', 5749),
        dropout_rate=config.get('dropout_rate', 0.5),
        use_batchnorm=config.get('use_batchnorm', True),
        use_global_avg_pool=config.get('use_global_avg_pool', False),
        use_residual=config.get('use_residual', False)
    )


def create_model(backbone_type, checkpoint_path=None, **model_kwargs):
    """
    Factory per creare modelli di face recognition.

    Args:
        backbone_type: "InceptionResnetV1" o "FaceEmbeddingCNN"
        checkpoint_path: path al .pth/.pt (opzionale, per caricare config da lì)
        **model_kwargs: parametri per costruire il modello (override della config del checkpoint)

    Returns:
        dict con:
            - 'model': istanza del modello PyTorch
            - 'image_size': dimensione input attesa (es. 160 o 128)
            - 'embedding_size': dimensione embedding output (es. 512, 128, 256)
            - 'use_get_embedding': True se il modello ha get_embedding(), False se usa forward()

    Examples:
        # InceptionResnetV1
        model_info = create_model("InceptionResnetV1")

        # FaceEmbeddingCNN con config da checkpoint
        model_info = create_model("FaceEmbeddingCNN", checkpoint_path="models/my_model.pth")

        # FaceEmbeddingCNN con config manuale
        model_info = create_model(
            "FaceEmbeddingCNN",
            checkpoint_path="models/my_model.pth",
            num_filters=[64, 128, 256],
            embedding_size=256
        )
    """

    if backbone_type == "InceptionResnetV1":
        model = InceptionResnetV1(pretrained=None)
        return {
            'model': model,
            'image_size': 160,
            'embedding_size': 512,
            'use_get_embedding': False  # InceptionResnetV1 usa forward()
        }

    elif backbone_type == "FaceEmbeddingCNN":
        # Se c'è un checkpoint, prova a caricare la config da lì
        config = {}

        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'config' in checkpoint:
                    config = checkpoint['config']
                    print(f" Config caricata da checkpoint:")
                    print(
                        f"  - Embedding size: {config.get('embedding_size', 'N/A')}")
                    print(
                        f"  - Num filters: {config.get('num_filters', 'N/A')}")
                    print(
                        f"  - Use residual: {config.get('use_residual', 'N/A')}")
            except Exception as e:
                print(
                    f" Warning: impossibile caricare config da checkpoint: {e}")

        # Override con parametri forniti dall'utente
        config.update(model_kwargs)

        # Valori di default se non specificati
        if not config:
            config = {
                'input_channels': 3,
                'num_filters': [32, 64, 128],
                'kernel_sizes': [3, 3, 3],
                'fc_hidden_size': 512,
                'embedding_size': 128,
                'dropout_rate': 0.5,
                'use_batchnorm': True,
                'use_global_avg_pool': False,
                'use_residual': False
            }
            print(" Warning: nessuna config trovata, uso valori di default")

        # Crea il modello
        model = create_cnn_model(config)

        return {
            'model': model,
            'image_size': 128,  # Le tue CNN usano 128x128
            'embedding_size': config.get('embedding_size', 128),
            'use_get_embedding': True  # FaceEmbeddingCNN ha get_embedding()
        }

    elif backbone_type == "FaceEmbeddingDNN":
        # DNN completamente connessa (per dimostrare diff rispetto alle CNN)
        config = {}

        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'config' in checkpoint:
                    config = checkpoint['config']
                    print(f" Config DNN caricata da checkpoint:")
                    print(
                        f"  - Embedding size: {config.get('embedding_size', 'N/A')}")
                    print(
                        f"  - Hidden sizes: {config.get('hidden_sizes', 'N/A')}")
            except Exception as e:
                print(
                    f" Warning: impossibile caricare config da checkpoint: {e}")

        # Override con parametri forniti
        config.update(model_kwargs)

        # Valori di default
        if not config:
            config = {
                'input_size': 128,
                'hidden_sizes': [2048, 1024, 512],
                'embedding_size': 128,
                'dropout_rate': 0.5,
                'use_batchnorm': True
            }
            print(" Warning: nessuna config DNN trovata, uso valori di default")

        # Crea il modello
        model = FaceEmbeddingDNN(**config)

        return {
            'model': model,
            'image_size': 128,  # DNN usa 128x128
            'embedding_size': config.get('embedding_size', 128),
            'use_get_embedding': True  # FaceEmbeddingDNN ha get_embedding()
        }

    else:
        raise ValueError(
            f"Backbone type '{backbone_type}' non supportato. "
            f"Usa 'InceptionResnetV1', 'FaceEmbeddingCNN' o 'FaceEmbeddingDNN'"
        )


def get_model_info(checkpoint_path):
    """
    Legge informazioni da un checkpoint .pth senza caricare il modello completo.
    Utile per debugging o per ispezionare i checkpoint.

    Args:
        checkpoint_path: percorso al file .pth

    Returns:
        dict con informazioni sul checkpoint

    Example:
        info = get_model_info("models/my_model.pth")
        print(f"Epoca: {info['epoch']}")
        print(f"Accuracy: {info['val_accuracy']}")
        print(f"Embedding size: {info['config']['embedding_size']}")
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if not isinstance(checkpoint, dict):
        return {
            'type': 'unknown',
            'message': 'Checkpoint non è un dizionario, probabilmente è un modello completo'
        }

    info = {
        'has_config': 'config' in checkpoint,
        'has_model_state': 'model_state_dict' in checkpoint,
        'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
    }

    if 'config' in checkpoint:
        info['config'] = checkpoint['config']

    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']

    if 'val_accuracy' in checkpoint:
        info['val_accuracy'] = checkpoint['val_accuracy']

    # Conta parametri nel model_state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        num_params = sum(p.numel() for p in state_dict.values())
        info['num_parameters'] = num_params
        info['state_dict_keys'] = list(state_dict.keys())[
            :10]  # Prime 10 chiavi

    return info
