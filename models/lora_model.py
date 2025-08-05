import tensorflow as tf
import keras_nlp as keras_hub  # Or use: import keras_nlp.models as keras_hub
from lora_layer import LoraLayer  # Assuming you've saved it as lora_layer.py

# Step 2: Set LoRA parameters
RANK = 4
ALPHA = 32.0
SEQ_LENGTH = 128
PRESET = "gpt2_base_en"

# Step 3: Load GPT-2 with preprocessor
preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
    PRESET,
    sequence_length=SEQ_LENGTH,
)
lora_model = keras_hub.models.GPT2CausalLM.from_preset(
    PRESET,
    preprocessor=preprocessor,
)

# Step 4: Apply LoRA to selected attention layers (query and value projections)
for layer_idx in range(lora_model.backbone.num_layers):
    decoder_layer = lora_model.backbone.get_layer(f"transformer_layer_{layer_idx}")
    self_attention_layer = decoder_layer._self_attention_layer

    # Allow modifications
    self_attention_layer._tracker.locked = False

    # Replace query and value dense layers with LoRA-wrapped versions
    self_attention_layer._query_dense = LoraLayer(
        self_attention_layer._query_dense,
        rank=RANK,
        alpha=ALPHA,
        trainable=True,
    )
    self_attention_layer._value_dense = LoraLayer(
        self_attention_layer._value_dense,
        rank=RANK,
        alpha=ALPHA,
        trainable=True,
    )

print("✅ LoRA layers successfully injected into GPT-2 model.")

output = lora_model(preprocessor(["LoRA is very useful for quick LLM finetuning"])[0])
print("✅ Inference successful")

for layer in lora_model._flatten_layers():
    lst_of_sublayers = list(layer._flatten_layers())

    if len(lst_of_sublayers) == 1:  # "leaves of the model"
        if layer.name in ["lora_A", "lora_B"]:
            layer.trainable = True
        else:
            layer.trainable = False

print("✅ Only LoRA layers are set as trainable")
