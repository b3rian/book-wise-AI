import math
import tensorflow as tf
from tensorflow import keras


class LoraLayer(keras.layers.Layer):
    """
    Implements the Low-Rank Adaptation (LoRA) mechanism by wrapping an existing Dense or EinsumDense layer.

    LoRA injects trainable low-rank matrices (A and B) into the original weight matrix of a frozen layer,
    allowing efficient fine-tuning of large pre-trained models with minimal trainable parameters.

    Attributes:
        original_layer (keras.layers.Layer): The frozen original dense or einsum layer.
        rank (int): The rank of the low-rank adapter matrices.
        alpha (int): Scaling factor to control the LoRA output magnitude.
        trainable (bool): Whether the LoRA adapters are trainable.
    """

    def __init__(
        self,
        original_layer: keras.layers.Layer,
        rank: int = 8,
        alpha: int = 32,
        trainable: bool = False,
        **kwargs,
    ):
        """
        Initializes the LoraLayer.

        Args:
            original_layer (keras.layers.Layer): The frozen Dense or EinsumDense layer to wrap.
            rank (int): The rank of the low-rank matrices A and B.
            alpha (int): Scaling factor applied to the LoRA output.
            trainable (bool): Whether to train the LoRA adapters.
            **kwargs: Additional keyword arguments passed to the base Layer class.
        """
        original_layer_config = original_layer.get_config()
        name = original_layer_config.get("name", "lora_wrapper")
        kwargs.pop("name", None)  # Avoid conflict with user-passed name

        super().__init__(name=name, trainable=trainable, **kwargs)

        self.rank = rank
        self.alpha = alpha
        self._scale = alpha / rank

        # Retrieve output shape and einsum equation (if applicable)
        self._equation = original_layer_config.get("equation")
        self._output_shape = original_layer_config.get("output_shape")

        # Save and freeze the original layer
        self.original_layer = original_layer
        self.original_layer.trainable = False  # Always frozen

        # Initialize LoRA adapter layers
        self.A = keras.layers.Dense(
            units=rank,
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            trainable=trainable,
            name=f"{name}_lora_A",
        )

        self.B = keras.layers.EinsumDense(
            equation=self._equation,
            output_shape=self._output_shape,
            kernel_initializer="zeros",
            trainable=trainable,
            name=f"{name}_lora_B",
        )

    def call(self, inputs, training=False):
        """
        Forward pass of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool): Whether the layer is in training mode.

        Returns:
            tf.Tensor: Output tensor with or without LoRA adaptation.
        """
        # Forward pass through the frozen original layer
        original_output = self.original_layer(inputs)

        if self.trainable and training:
            # Compute LoRA output and add it to the original output
            lora_output = self.B(self.A(inputs)) * self._scale
            return original_output + lora_output

        # During inference or if LoRA is not trainable
        return original_output
