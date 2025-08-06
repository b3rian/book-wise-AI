"""
Main training script for training and evaluating a Keras model using a Trainer wrapper.

Features:
- Loads training configuration from YAML
- Sets reproducible seeds
- Loads training, validation, and test datasets
- Initializes and trains the model
- Saves final model and evaluates it on val/test sets
"""

import os
import yaml
import tensorflow as tf
from training.trainer import Trainer
from utils.logging import get_callbacks
from data.prerocessing_pipeline import main
from utils.seed import set_seed
from models.transformer_decoder_model import create_model
from configs import config

# Set random seed for full reproducibility
set_seed = config.training["seed"]

#Load training, validation, and test datasets
data_dir = config["dataset"]["data_dir"]
batch_size = config["dataset"]["batch_size"]

train_ds, val_ds, test_ds = main()

# Define model function that returns a compiled model
def model_fn():
    return create_model(
        maxlen=config.model["max_sequence_length"],
        vocab_size=config.model["vocab_size"],
        embed_dim=config.model["embed_dim"],
        num_heads=config.model["num_heads"],
        feed_forward_dim=config.model["feed_forward_dim"]
    )

# Initialize Trainer and start training
trainer = Trainer(
    model_fn=model_fn,
    train_ds=train_ds,
    val_ds=val_ds,
    config=config
)

model = trainer.train()

# Evaluate final model on validation and test datasets
print("\n✅ Evaluating model on validation set...")
val_loss, val_acc = model.evaluate(val_ds)
print(f"📊 Final Validation Accuracy: {val_acc:.4f}")

print("\n🧪 Evaluating model on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"🧪 Test Accuracy: {test_acc:.4f}")

# Save final trained model to disk
os.makedirs("exports", exist_ok=True)
model_path = "exports/transformer_decoder_model.keras"
model.save(model_path)
print(f"\n✅ Final model saved to: {model_path}")