import tensorflow as tf
from tensorflow import keras

# Constants (ensure these are defined beforehand)
ALPHA = 32
RANK = 8
EPOCHS = 3  # Or your preferred value
train_ds = ...  # Your preprocessed dataset
gpu_memory_callback = ...  # Optional memory monitor callback

# -------------------------------
# Optimizer and Loss Setup
# -------------------------------
def get_optimizer_and_loss():
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
        epsilon=1e-6,
        global_clipnorm=1.0,  # Gradient clipping
    )
    # Exclude LayerNorm and bias terms from weight decay
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return optimizer, loss

optimizer, loss = get_optimizer_and_loss()

# -------------------------------
# Compile LoRA Model
# -------------------------------
lora_model.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

# -------------------------------
# Train Model
# -------------------------------
lora_model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[gpu_memory_callback],  # Optional
)

# -------------------------------
# Merge LoRA Weights Into Base Model
# -------------------------------
for layer_idx in range(lora_model.backbone.num_layers):
    decoder_layer = lora_model.backbone.get_layer(f"transformer_layer_{layer_idx}")
    self_attention_layer = decoder_layer._self_attention_layer

    # --- Merge query dense layer ---
    query_lora = self_attention_layer._query_dense
    A = query_lora.A.kernel  # (hidden_dim, rank)
    B = query_lora.B.kernel  # (rank, num_heads, head_dim)
    merged_weights = tf.einsum("ab,bcd->acd", A, B) * (ALPHA / RANK)
    query_lora.original_layer.kernel.assign_add(merged_weights)

    # Replace with original layer (now updated)
    self_attention_layer._query_dense = query_lora.original_layer

    # --- Merge value dense layer ---
    value_lora = self_attention_layer._value_dense
    A = value_lora.A.kernel
    B = value_lora.B.kernel
    merged_weights = tf.einsum("ab,bcd->acd", A, B) * (ALPHA / RANK)
    value_lora.original_layer.kernel.assign_add(merged_weights)

    # Replace with original layer
    self_attention_layer._value_dense = value_lora.original_layer

print("✅ LoRA weights merged into base model successfully.")

# Evaluate model
test_loss, test_accuracy = lora_model.evaluate(test_ds)
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

import os

# Create export directory
export_dir = "exports2"
os.makedirs(export_dir, exist_ok=True)

# Save the model
lora_model.save(os.path.join(export_dir, "gpt2_lora_merged.keras"))
print("✅ Model saved to 'exports/gpt2_lora_merged.keras'")
