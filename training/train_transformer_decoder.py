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
from trainers.trainer import Trainer
from utils.logger import get_callbacks
from data.input_pipeline import get_datasets
from utils.seed import set_seed
from models.build_vgg_19 import simple_cnn_tiny_imagenet

# -------------------------------------------------------------------------
# Step 1: Load configurations from YAML
# -------------------------------------------------------------------------
CONFIG_PATH = "configs/resnet.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# -------------------------------------------------------------------------
# Step 2: Set random seed for full reproducibility
# -------------------------------------------------------------------------
set_seed(config["seed"])

# -------------------------------------------------------------------------
# Step 3: Load training, validation, and test datasets
# -------------------------------------------------------------------------
data_dir = config["dataset"]["data_dir"]
batch_size = config["dataset"]["batch_size"]

train_ds, val_ds, test_ds = get_datasets(data_dir, batch_size)

# -------------------------------------------------------------------------
# Step 4: Define model function that returns a compiled model
# -------------------------------------------------------------------------
def model_fn():
    return simple_cnn_tiny_imagenet(
        input_shape=(64, 64, 3),
        num_classes=config["model"]["num_classes"]
    )

# -------------------------------------------------------------------------
# Step 5: Initialize Trainer and start training
# -------------------------------------------------------------------------
trainer = Trainer(
    model_fn=model_fn,
    train_ds=train_ds,
    val_ds=val_ds,
    config=config
)

model = trainer.train()

# -------------------------------------------------------------------------
# Step 6: Evaluate final model on validation and test datasets
# -------------------------------------------------------------------------
print("\n✅ Evaluating model on validation set...")
val_loss, val_acc = model.evaluate(val_ds)
print(f"📊 Final Validation Accuracy: {val_acc:.4f}")

print("\n🧪 Evaluating model on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"🧪 Test Accuracy: {test_acc:.4f}")

# -------------------------------------------------------------------------
# Step 7: Save final trained model to disk
# -------------------------------------------------------------------------
os.makedirs("exports", exist_ok=True)
model_path = "exports/custom_model.keras"
model.save(model_path)
print(f"\n✅ Final model saved to: {model_path}")
