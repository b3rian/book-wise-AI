from tensorflow import keras
from keras import layers, ops
from models.transformer_decoder_block import TransformerBlock
from models.embeddings import TokenAndPositionEmbedding
from configs import config

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
    # Access model hyperparameters
    maxlen = config.model["max_sequence_length"]
    vocab_size = config.model["vocab_size"]
    embed_dim = config.model["embed_dim"]
    num_heads = config.model["num_heads"]
    ff_dim = config.model["feed_forward_dim"]
    num_layers = config.model.get("num_transformer_blocks", 4)
    
    # Input layer expecting integer token IDs
    inputs = layers.Input(shape=(maxlen,), dtype="int32", name="input_tokens")

    # Token and position embeddings
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    # Transformer block with causal masking
    # Stack Transformer blocks dynamically
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # Final dense layer maps to vocabulary size for language modeling
    logits = layers.Dense(vocab_size, name="output_logits")(x)

    # Define model with both logits and intermediate embeddings as output (for optional use)
    model = keras.Model(inputs=inputs, outputs=[logits, x], name="transformer_decoder")

    return model