import os
import csv
import argparse
from typing import List, Tuple

import librosa
import numpy as np
import tensorflow_hub as hub
from tensorflow import keras


YAMNET_SR = 16000


def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")


def list_audio_files(root: str) -> List[Tuple[str, str]]:
    """
    Walk training_data/ and return (filepath, label_name) pairs.

    Expected layout:
        training_data/
            KICK_BASS/*.wav
            SNARE_CLAP/*.wav
            HIHAT_CYMBAL/*.wav
            BASS/*.wav
            GUITAR/*.wav
            KEYS/*.wav
            STRINGS/*.wav
            LEAD_VOCAL/*.wav
            OTHER/*.wav
    """
    items: List[Tuple[str, str]] = []
    for label in sorted(os.listdir(root)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for name in os.listdir(label_dir):
            if not name.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                continue
            items.append((os.path.join(label_dir, name), label))
    return items


def extract_embeddings(
    yamnet, items: List[Tuple[str, str]]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute averaged YAMNet embeddings and label indices."""
    label_names: List[str] = sorted({label for _, label in items})
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    xs: List[np.ndarray] = []
    ys: List[int] = []

    for path, label in items:
        y, _ = librosa.load(path, sr=YAMNET_SR, mono=True)
        y = y.astype(np.float32)
        _, embeddings, _ = yamnet(y)
        emb = np.array(embeddings)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        xs.append(emb)
        ys.append(label_to_idx[label])

    return np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.int64), label_names


def build_classifier(input_dim: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="embedding")
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="stem_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train stem classifier on YAMNet embeddings.")
    parser.add_argument(
        "--data-dir",
        default="training_data",
        help="Root folder with per-label subfolders of audio files.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save stem_classifier model and labels.",
    )
    args = parser.parse_args()

    items = list_audio_files(args.data_dir)
    if not items:
        raise SystemExit(
            f"No training audio found under {args.data_dir}. "
            "Expected structure: training_data/LABEL_NAME/*.wav"
        )

    print(f"Found {len(items)} training files across labels:")
    by_label = {}
    for _, label in items:
        by_label[label] = by_label.get(label, 0) + 1
    for label, count in by_label.items():
        print(f"  {label}: {count}")

    print("Loading YAMNet from TensorFlow Hub...")
    yamnet = load_yamnet()

    print("Extracting embeddings...")
    X, y, label_names = extract_embeddings(yamnet, items)
    num_classes = len(label_names)
    input_dim = X.shape[1]

    print(f"Feature matrix: {X.shape}, num_classes={num_classes}")

    # Simple train/validation split
    num_samples = X.shape[0]
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    split = int(num_samples * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = build_classifier(input_dim, num_classes)
    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        verbose=1,
    )

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "stem_classifier.keras")
    labels_path = os.path.join(args.models_dir, "stem_labels.txt")

    print(f"Saving model to {model_path}")
    model.save(model_path)

    print(f"Saving label names to {labels_path}")
    with open(labels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for name in label_names:
            writer.writerow([name])

    print("Done.")


if __name__ == "__main__":
    main()

