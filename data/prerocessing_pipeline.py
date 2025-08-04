import os
import tensorflow as tf
import keras_nlp as keras_hub 
from tensorflow.keras import layers

# constants
SEQ_LEN = 128 # Maximum sequence length for tokenization
BATCH_SIZE = 64 # Batch size for training
AUTOTUNE = tf.data.AUTOTUNE # Auto-tune dataset performance

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
    
    # Create a StartEndPacker layer to handle start token and padding
    start_packer = keras_hub.layers.StartEndPacker(
        sequence_length=seq_len,
        start_value=tokenizer.token_to_id("[BOS]"),
    )

    return tokenizer, start_packer

def preprocess_fn(text, tokenizer, start_packer):
    """
    Tokenizes and packs input text for training.

    Args:
        text (tf.Tensor): Raw text input.
        tokenizer: WordPiece tokenizer.
        start_packer: Layer to pack and add [BOS] token.

    Returns:
        tuple: (input_tensor, label_tensor)
    """
    tokens = tokenizer(text)
    inputs = start_packer(tokens)
    labels = tokens  # Model learns to predict next tokens
    return inputs, labels

def create_dataset(file_path, tokenizer, start_packer, is_training=False):
    """
    Create a tf.data.Dataset pipeline.

    Args:
        file_path (str): Path to the cleaned dataset file.
        tokenizer: Tokenizer instance.
        start_packer: Token packer layer.
        is_training (bool): Whether the dataset is used for training.

    Returns:
        tf.data.Dataset: Preprocessed batched dataset.
    """
    ds = tf.data.TextLineDataset(file_path) # Load text lines from file

    if is_training:
        ds = ds.cache().shuffle(10000) # Shuffle and cache dataset for training

    ds = (
        ds.map(lambda x: preprocess_fn(x, tokenizer, start_packer), num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE) # Batch the dataset
          .prefetch(AUTOTUNE) # Prefetch for performance
    )
    return ds

def main():
    # Base directory containing raw and vocab files
    raw_data_dir = "/kaggle/input/simple-books/simplebooks/simplebooks-2-raw"
    clean_data_dir = raw_data_dir  # Reusing the same directory
    vocab_path = "/kaggle/input/simple-books/simplebooks/simplebooks-92/train.vocab"

    # Define file paths
    train_raw = os.path.join(raw_data_dir, "train.txt")
    valid_raw = os.path.join(raw_data_dir, "valid.txt")
    test_raw = os.path.join(raw_data_dir, "test.txt")

    # Cleaned file paths
    train_clean = os.path.join(clean_data_dir, "train_clean.txt")
    valid_clean = os.path.join(clean_data_dir, "valid_clean.txt")
    test_clean = os.path.join(clean_data_dir, "test_clean.txt")

    # Clean and save text
    write_cleaned_lines(train_clean, load_and_clean_lines(train_raw))
    write_cleaned_lines(valid_clean, load_and_clean_lines(valid_raw))
    write_cleaned_lines(test_clean, load_and_clean_lines(test_raw))

    # Build tokenizer and packer
    tokenizer, start_packer = build_tokenizer(vocab_path, seq_len=SEQ_LEN)

    # Create datasets
    train_ds = create_dataset(train_clean, tokenizer, start_packer, is_training=True)
    val_ds = create_dataset(valid_clean, tokenizer, start_packer)
    test_ds = create_dataset(test_clean, tokenizer, start_packer)

    return train_ds, val_ds, test_ds

# Run the preprocessing pipeline
train_ds, val_ds, test_ds = main()