# train_model.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
IMG_SIZE = (160, 160)
BATCH = 16
EPOCHS = 30

MODEL_OUT = os.path.join(BASE_DIR, "gesture_classifier.keras")  # Keras native format
MAPPING_OUT = os.path.join(BASE_DIR, "class_indices.json")
CHECKPOINT = os.path.join(BASE_DIR, "best_gesture.keras")

if not os.path.isdir(DATA_DIR):
    raise SystemExit("dataset/ not found — run collect_data.py first")

# -----------------------------
# Data generators with augmentation
# -----------------------------
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

datagen = ImageDataGenerator(
    preprocessing_function=preprocess,
    validation_split=0.2,
    rotation_range=22,
    width_shift_range=0.18,
    height_shift_range=0.18,
    zoom_range=0.18,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True,
    fill_mode="reflect"
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Train/Val samples:", train_gen.samples, val_gen.samples)
print("Class indices:", train_gen.class_indices)

# save class mapping
with open(MAPPING_OUT, "w") as f:
    json.dump(train_gen.class_indices, f)

# class weights to handle imbalance
classes = train_gen.classes
unique, counts = np.unique(classes, return_counts=True)
class_counts = dict(zip(unique, counts))
total = classes.shape[0]
class_weight = {}
for k, v in class_counts.items():
    class_weight[int(k)] = float(total) / (len(class_counts) * v)
print("Class weights:", class_weight)

# -----------------------------
# Build model (MobileNetV2 transfer)
# -----------------------------
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base.trainable = False

inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.GaussianNoise(0.08)(inp)
x = base(x, training=False)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inp, out)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

cb_list = [
    callbacks.ModelCheckpoint(CHECKPOINT, monitor="val_loss", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6)
]

# train frozen base
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=cb_list
)

# fine-tune
print("Fine-tuning...")
base.trainable = True
fine_tune_at = len(base.layers) - 40
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ft_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    class_weight=class_weight,
    callbacks=cb_list
)

# save final model in Keras native format (avoids old h5 issues)
model.save(MODEL_OUT)
print("Saved model to", MODEL_OUT)
print("Saved mapping to", MAPPING_OUT)
