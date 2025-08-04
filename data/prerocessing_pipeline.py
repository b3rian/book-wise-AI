import os
import tensorflow as tf
import keras_nlp as keras_hub
from tensorflow.keras import layers

# constants
SEQ_LEN = 128
BATCH_SIZE = 64

def load_and_clean_lines(file_path, min_words=3, max_words=250):
    """
    Load and clean lines from a given text file.

    Args:
        file_path (str): Path to the text file.
        min_words (int): Minimum number of words per line.
        max_words (int): Maximum number of words per line.

    Returns:
        list: A list of cleaned text lines.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    return [
        line.strip()
        for line in lines
        if line.strip() and min_words < len(line.strip().split()) < max_words
    ]

def write_cleaned_lines(output_path, lines):
    """
    Write cleaned lines to a text file.

    Args:
        output_path (str): Destination file path.
        lines (list): List of cleaned strings.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("/n".join(lines))


def build_tokenizer(vocab_path, seq_len=128):
    """
    Build WordPiece tokenizer and packing layer.

    Args:
        vocab_path (str): Path to vocabulary file.
        seq_len (int): Maximum sequence length.

    Returns:
        tokenizer: A WordPiece tokenizer.
        start_packer: A layer that adds a start token and pads/truncates to `seq_len`.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip().split("/t")[0] for line in f if line.strip()]

     # Add special tokens
    reserved_tokens = ["[PAD]", "[UNK]", "[BOS]"]
    vocab = reserved_tokens + vocab

    tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        sequence_length=seq_len,
        lowercase=False,
    )