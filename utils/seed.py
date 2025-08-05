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