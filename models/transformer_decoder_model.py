from tensorflow import keras
from keras import layers, ops
from models.transformer_decoder_block import TransformerBlock
from models.embeddings import TokenAndPositionEmbedding

def create_model(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim):
    """
    Builds and compiles a simple transformer-based language model.

    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of token and position embeddings.
        num_heads (int): Number of attention heads in the transformer block.
        feed_forward_dim (int): Dimension of the feed-forward network.

    Returns:
        keras.Model: Compiled Keras model ready for training.
    """