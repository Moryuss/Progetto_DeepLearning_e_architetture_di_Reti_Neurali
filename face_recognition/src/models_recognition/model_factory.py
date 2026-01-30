import torch
from facenet_pytorch import InceptionResnetV1
from .face_embedding_cnn import FaceEmbeddingCNN
from .face_embedding_dnn import FaceEmbeddingDNN
from .face_embedding_TransferLearning_and_FineTuning import TransferLearningEmbeddingCNN


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
                    # FIX: Gestisci config annidata (model_config)
                    checkpoint_config = checkpoint['config']

                    # Se la config ha 'model_config' annidato, estrailo
                    if 'model_config' in checkpoint_config:
                        config = checkpoint_config['model_config'].copy()
                        print(
                            f"✓ Config caricata da checkpoint['config']['model_config']:")
                    else:
                        # Altrimenti usa direttamente 'config' (compatibilità)
                        config = checkpoint_config.copy()
                        print(f"✓ Config caricata da checkpoint['config']:")

                    print(
                        f"  - Embedding size: {config.get('embedding_size', 'N/A')}")
                    print(
                        f"  - Num filters: {config.get('num_filters', 'N/A')}")
                    print(
                        f"  - Use residual: {config.get('use_residual', 'N/A')}")
                    print(
                        f"  - Use GAP: {config.get('use_global_avg_pool', 'N/A')}")
                    print(f"  - Dropout: {config.get('dropout_rate', 'N/A')}")
                else:
                    print(" Checkpoint non contiene 'config', uso valori di default")
            except Exception as e:
                print(
                    f" Warning: impossibile caricare config da checkpoint: {e}")

        # Override con parametri forniti dall'utente (hanno priorità massima)
        config.update(model_kwargs)

        # Valori di default SOLO se config è completamente vuota
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
            print("⚠️ Warning: nessuna config trovata, uso valori di default")

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
                    # FIX: Gestisci config annidata anche per DNN
                    checkpoint_config = checkpoint['config']

                    if 'model_config' in checkpoint_config:
                        config = checkpoint_config['model_config'].copy()
                        print(
                            f"✓ Config DNN caricata da checkpoint['config']['model_config']:")
                    else:
                        config = checkpoint_config.copy()
                        print(
                            f"✓ Config DNN caricata da checkpoint['config']:")

                    print(
                        f"  - Embedding size: {config.get('embedding_size', 'N/A')}")
                    print(
                        f"  - Hidden sizes: {config.get('hidden_sizes', 'N/A')}")
                else:
                    print(
                        "⚠️ Checkpoint DNN non contiene 'config', uso valori di default")
            except Exception as e:
                print(
                    f"⚠️ Warning: impossibile caricare config DNN da checkpoint: {e}")

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
            print("⚠️ Warning: nessuna config DNN trovata, uso valori di default")

        # Crea il modello
        model = FaceEmbeddingDNN(**config)

        return {
            'model': model,
            'image_size': 128,  # DNN usa 128x128
            'embedding_size': config.get('embedding_size', 128),
            'use_get_embedding': True  # FaceEmbeddingDNN ha get_embedding()
        }

    elif backbone_type == 'TransferLearningEmbeddingCNN':
        # Transfer Learning CNN con backbone pre-addestrato (ResNet)
        config = {}

        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'config' in checkpoint:
                    # Gestisci config annidata (model_config)
                    checkpoint_config = checkpoint['config']

                    if 'model_config' in checkpoint_config:
                        config = checkpoint_config['model_config'].copy()
                        print(
                            f"✓ Config Transfer Learning caricata da checkpoint['config']['model_config']:")
                    else:
                        config = checkpoint_config.copy()
                        print(
                            f"✓ Config Transfer Learning caricata da checkpoint['config']:")

                    print(
                        f"  - Embedding size: {config.get('embedding_size', 'N/A')}")
                    print(
                        f"  - ResNet version: {config.get('resnet_version', 'N/A')}")
                    print(
                        f"  - Freeze backbone: {config.get('freeze_resnet_backbone', 'N/A')}")
                    print(
                        f"  - Num filters: {config.get('num_filters', 'N/A')}")
                    print(
                        f"  - Use GAP: {config.get('use_global_avg_pool', 'N/A')}")
                    print(f"  - Dropout: {config.get('dropout_rate', 'N/A')}")
                else:
                    print(
                        "⚠️ Checkpoint Transfer Learning non contiene 'config', uso valori di default")
            except Exception as e:
                print(
                    f"⚠️ Warning: impossibile caricare config Transfer Learning da checkpoint: {e}")

        # Override con parametri forniti dall'utente
        config.update(model_kwargs)

        # Valori di default
        if not config:
            config = {
                'use_pretrained_resnet': True,
                'resnet_version': 'resnet18',
                'freeze_resnet_backbone': False,
                'num_filters': [256, 256],
                'kernel_sizes': [3, 3],
                'fc_hidden_size': 512,
                'embedding_size': 128,
                'dropout_rate': 0.3,
                'use_batchnorm': True,
                'use_global_avg_pool': True
            }
            print(
                "⚠️ Warning: nessuna config Transfer Learning trovata, uso valori di default")

        # Crea il modello
        model = TransferLearningEmbeddingCNN(**config)

        return {
            'model': model,
            'image_size': 224,  # ResNet usa tipicamente 224x224
            'embedding_size': config.get('embedding_size', 128),
            'use_get_embedding': True  # Ha get_embedding()
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
        #  Gestisci config annidata anche qui
        checkpoint_config = checkpoint['config']
        if 'model_config' in checkpoint_config:
            info['model_config'] = checkpoint_config['model_config']
            info['training_config'] = checkpoint_config.get(
                'training_config', {})
        else:
            info['config'] = checkpoint_config

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

        #  Deduce num_filters dallo state_dict
        if 'conv_blocks.0.weight' in state_dict:
            try:
                f1 = state_dict['conv_blocks.0.weight'].shape[0]
                f2 = state_dict['conv_blocks.5.weight'].shape[0]
                f3 = state_dict['conv_blocks.10.weight'].shape[0]
                info['detected_num_filters'] = [f1, f2, f3]
            except:
                pass

    return info
