import os
import tensorflow as tf
import keras_nlp as keras_hub
from tensorflow.keras import layers

# constants
SEQ_LEN = 128
BATCH_SIZE = 64