from tensorflow import keras
from keras import ops
from keras import layers

class TokenAndPositionEmbedding(layers.Layer):
    """
    Combines token and positional embeddings.

    This layer learns:
    - An embedding vector for each token in the vocabulary.
    - An embedding vector for each position in the input sequence.
    
    The final embedding is a sum of the token embedding and the positional embedding.
    """
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int):
        """
        Initializes the token and position embedding layers.

        Args:
            maxlen (int): Maximum length of the input sequences.
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    
    def call(self, x):
        """
        Applies token and positional embeddings to the input.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, sequence_length, embed_dim).
        """