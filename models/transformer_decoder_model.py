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
    # Input layer expecting integer token IDs
    inputs = layers.Input(shape=(maxlen,), dtype="int32", name="input_tokens")

    # Token and position embeddings
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    # Transformer block with causal masking
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)

    # Final dense layer maps to vocabulary size for language modeling
    logits = layers.Dense(vocab_size, name="output_logits")(x)

    # Define model with both logits and intermediate embeddings as output (for optional use)
    model = keras.Model(inputs=inputs, outputs=[logits, x], name="transformer_decoder")

    # Use sparse categorical cross-entropy with logits for loss calculation
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compile model - we only optimize based on the first output (logits)
    model.compile(
        optimizer="adam",
        loss=[loss_fn, None],  # No loss applied to the intermediate output `x`
    )

    return model