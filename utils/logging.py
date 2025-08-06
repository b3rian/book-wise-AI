import os
from datetime import datetime
import tensorflow as tf


def get_callbacks(
    base_dir: str = "experiments",
    monitor: str = "val_loss",
    model_name: str = "transformer_decoder_model"
) -> list:
    """
    Creates standard Keras callbacks for training monitoring, checkpointing, and early stopping.

    Args:
        base_dir (str): Directory where experiment logs and checkpoints are stored.
        monitor (str): Metric to monitor for checkpointing, LR reduction, and early stopping.
        model_name (str): Name of the model used in checkpoint filename.

    Returns:
        list: A list of tf.keras.callbacks.Callback instances.
    """
    # Timestamp for experiment versioning
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, timestamp)

    # Paths
    log_dir = os.path.join(experiment_dir, "logs")
    ckpt_path = os.path.join(experiment_dir, "checkpoints", f"best_{model_name}.keras")
    csv_log_path = os.path.join(experiment_dir, "metrics.csv")

    # Ensure directories exist
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create callbacks
    return [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=csv_log_path,
            append=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]
