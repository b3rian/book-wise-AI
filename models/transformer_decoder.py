import tensorflow as tf
from tensorflow import keras
from keras import layers, ops

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Generates a causal attention mask to prevent attention to future tokens.

    This is used in decoder-only architectures like GPT, where tokens should
    only attend to previous or current positions (not future ones).

    Parameters:
    - batch_size (int or Tensor): Number of sequences in a batch.
    - n_dest (int): Number of destination positions (usually equal to sequence length).
    - n_src (int): Number of source positions (same as n_dest for self-attention).
    - dtype (tf.DType or str): The data type of the output mask, e.g., 'bool' or 'float32'.

    Returns:
    - tf.Tensor: A lower triangular mask of shape (batch_size, n_dest, n_src)
    """
    # Create destination and source position indices
    i = ops.arange(n_dest)[:, None]  # Shape: (n_dest, 1)
    j = ops.arange(n_src)            # Shape: (n_src,)

    # Compute lower triangular matrix (causal mask)
    mask_matrix = i >= j - n_src + n_dest
    mask = ops.cast(mask_matrix, dtype)  # Convert boolean mask to specified dtype

    # Reshape to add batch dimension
    mask = ops.reshape(mask, [1, n_dest, n_src])

    # Tile the mask to match the batch size
    mult = ops.concatenate([
        ops.expand_dims(batch_size, -1),  # Shape: [1]
        ops.convert_to_tensor([1, 1])     # Shape: [2]
    ], axis=0)
    
    return ops.tile(mask, mult)  # Final shape: (batch_size, n_dest, n_src)

class TransformerBlock(layers.Layer):
    """
    A single transformer decoder block implementing:
    - Causal self-attention (no lookahead)
    - Feedforward neural network (FFN)
    - Residual connections
    - Layer normalization
    - Dropout for regularization
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, name=None):
        """
        Initializes the transformer block.

        Parameters:
        - embed_dim (int): Dimension of the token embeddings.
        - num_heads (int): Number of attention heads.
        - ff_dim (int): Hidden dimension of the feedforward network.
        - rate (float): Dropout rate.
        - name (str): Optional name for the layer.
        """
        super().__init__(name=name) # Initialize the base Layer class

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),  # Position-wise feedforward
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        """
        Executes the forward pass of the transformer block.

        Parameters:
        - inputs (tf.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
        - training (bool): Whether the call is in training mode (enables dropout)

        Returns:
        - tf.Tensor: Output tensor of same shape as input
        """
