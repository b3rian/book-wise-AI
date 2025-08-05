"""Trainer class for managing the training process of a Keras model."""

import tensorflow as tf
from utils.metrics import get_classification_metrics
from utils.logger import get_callbacks
import keras

class Trainer:
    def __init__(
        self,
        model_fn: callable,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        config: dict
    ):
        """
        Initialize the Trainer.

        Args:
            model_fn (Callable): Function that returns a Keras model instance.
            train_ds (tf.data.Dataset): Prepared training dataset.
            val_ds (tf.data.Dataset): Prepared validation dataset.
            config (dict): Dictionary loaded from YAML config file.
        """
        self.config = config
        self.model = model_fn()           # Build the model
        self._compile_model()             # Compile with optimizer, loss, metrics
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.callbacks = get_callbacks(   # Initialize callbacks
            monitor=config["training"].get("monitor", "val_loss")
        )

    def _compile_model(self):
        """
        Compile the model using settings from the configuration.
        Supports Adam and SGD optimizers with optional weight decay.
        """
        training_cfg = self.config["training"]
        optimizer_cfg = training_cfg["optimizer"]
        opt_name = optimizer_cfg["name"].lower()
        lr = training_cfg["learning_rate"]["initial"]
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Create optimizer
        if opt_name == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=optimizer_cfg.get("beta1", 0.9),
                beta_2=optimizer_cfg.get("beta2", 0.999),
                weight_decay=training_cfg.get("weight_decay", 0.0)
            )
        elif opt_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=optimizer_cfg.get("momentum", 0.9),
                weight_decay=training_cfg.get("weight_decay", 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=get_classification_metrics()
        )

    def train(self) -> tf.keras.Model:
        """
        Execute the model training loop.

        Returns:
            Trained Keras model.
        """
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config["training"]["epochs"],
            verbose=1,
            callbacks=self.callbacks
        )
