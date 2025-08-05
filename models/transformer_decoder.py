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