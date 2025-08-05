"""
Utility module for setting global random seeds across Python, NumPy, and TensorFlow
to ensure reproducibility in machine learning experiments.
"""

import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 42) -> None:
    """
    Set global random seed for Python, NumPy, and TensorFlow for reproducibility.

    Args:
        seed (int): The seed value to use for random number generators. Default is 42.

    Usage:
        >>> from utils.seed import set_seed
        >>> set_seed(123)
    """
    
    # Set Python built-in randomness seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set TensorFlow randomness
    tf.random.set_seed(seed)

    # Keras-specific additional seed setting for deterministic initialization
    tf.keras.utils.set_random_seed(seed)

    # Enable full determinism in TensorFlow operations
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        print("[WARNING] `enable_op_determinism` not available in this TensorFlow version.")

    print(f"[INFO] Global seed set to {seed}")