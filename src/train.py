"""Train and evaluate the proposed CNN.

Expected folder layout: <data_dir>/{Train,Valid,Test}/{Early_Blight,Healthy,Late_Blight}/*.jpg
Usage: python src/train.py --data_dir data/merged-augmented
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model


def make_gen(path, shuffle, seed=None):
    return ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        path,
        target_size=(256, 256),
        batch_size=32,
        class_mode="categorical",
        shuffle=shuffle,
        seed=seed,
    )


def main(data_dir, epochs, out):
    train = make_gen(f"{data_dir}/Train", True, 42)
    valid = make_gen(f"{data_dir}/Valid", False)
    test = make_gen(f"{data_dir}/Test", False)

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        EarlyStopping(
            monitor="val_loss", patience=10, min_delta=0.001, restore_best_weights=True
        ),
        ModelCheckpoint(filepath=out, monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(train, epochs=epochs, validation_data=valid, callbacks=callbacks)

    pred = np.argmax(model.predict(test), axis=1)
    labels = list(test.class_indices.keys())
    print(classification_report(test.classes, pred, target_names=labels))
    sns.heatmap(
        confusion_matrix(test.classes, pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default="best_model.keras")
    a = ap.parse_args()
    main(a.data_dir, a.epochs, a.out)
