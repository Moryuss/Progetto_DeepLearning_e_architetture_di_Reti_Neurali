import torch.nn as nn


class FaceEmbeddingDNN(nn.Module):
    """
    Deep Neural Network per generare embeddings di volti.
    Durante training: classificazione delle identità (cross entropy)
    Durante inference: estrazione embedding dal penultimo layer
    """

    def __init__(self,
                 input_size=128*128*3,  # immagini 128x128 RGB flattened
                 hidden_sizes=[1024, 512],  # dimensioni layer nascosti
                 embedding_size=128,  # dimensione embedding finale
                 num_classes=5749,  # numero identità in DataSet LFW
                 dropout_rate=0.5,  # dropout rate
                 use_batchnorm=True):  # usa batch normalization
        """
        Args:
            input_size: dimensione input (img_height * img_width * channels)
            hidden_sizes: lista con dimensioni dei layer nascosti
            embedding_size: dimensione del vettore embedding
            num_classes: numero di identità (classi) per classificazione
            dropout_rate: probabilità dropout (0 = no dropout)
            use_batchnorm: se True, aggiunge BatchNorm dopo ogni layer
        """
        super(FaceEmbeddingDNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.use_batchnorm = use_batchnorm

        # Costruzione layers sequenziali per feature extraction
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Embedding layer (penultimo layer - questo è ciò che useremo per embeddings)
        layers.append(nn.Linear(prev_size, embedding_size))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(embedding_size))
        layers.append(nn.ReLU())

        # Sequential usa i layer che abbiamo appena fatto append. Di solito sarebbe sequential (layer1, layer2 ...)
        self.feature_extractor = nn.Sequential(*layers)

        # Classification head (solo per training, rimosso durante inference)
        # last alyer fully conencted, da dim ultimo hidden layer a dim clssi
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        """
        Forward pass completo per training (con classificazione)

        Args:
            x: tensor [batch_size, channels, height, width]

        Returns:
            logits: tensor [batch_size, num_classes]
        """
        # Flatten dell'immagine
        x = x.view(x.size(0), -1)  # [batch_size, input_size]

        # Estrazione features + embedding
        embedding = self.feature_extractor(x)  # [batch_size, embedding_size]

        # Classificazione
        logits = self.classifier(embedding)  # [batch_size, num_classes]

        return logits

    def get_embedding(self, x):
        """
        Estrae solo l'embedding (per inference/confronto facce)

        Args:
            x: tensor [batch_size, channels, height, width]

        Returns:
            embedding: tensor [batch_size, embedding_size]
        """
        # Flatten dell'immagine
        x = x.view(x.size(0), -1)

        # Estrazione embedding (senza classification head, senza classifier)
        embedding = self.feature_extractor(x)

        return embedding
