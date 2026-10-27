"""Proposed 6-block CNN for potato leaf disease classification (from pldd-sequential notebook)."""

import tensorflow as tf
from tensorflow.keras import layers, models

L2 = tf.keras.regularizers.l2


def conv_block(filters, l2):
    return [
        layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer=L2(l2)),
        layers.LeakyReLU(negative_slope=0.01),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
    ]


def build_model(input_shape=(256, 256, 3), num_classes=3):
    stack = [layers.InputLayer(shape=input_shape)]
    for filters, l2 in [
        (32, 1e-4),
        (64, 1.5e-4),
        (128, 2e-4),
        (256, 2.5e-4),
        (512, 3e-4),
        (1024, 3.5e-4),
    ]:
        stack += conv_block(filters, l2)
    stack += [
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu", kernel_regularizer=L2(4e-4)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ]
    return models.Sequential(stack)


if __name__ == "__main__":
    build_model().summary()
