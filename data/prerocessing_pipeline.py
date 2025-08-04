import os
import tensorflow as tf
import keras_nlp as keras_hub
from tensorflow.keras import layers

# constants
SEQ_LEN = 128
BATCH_SIZE = 64

def load_and_clean_lines(file_path, min_words=5, max_words=250):
    """
    Load and clean lines from a given text file.

    Args:
        file_path (str): Path to the text file.
        min_words (int): Minimum number of words per line.
        max_words (int): Maximum number of words per line.

    Returns:
        list: A list of cleaned text lines.
    """