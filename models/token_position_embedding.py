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