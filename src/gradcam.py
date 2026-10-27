"""Grad-CAM for the proposed CNN.

Usage: python src/gradcam.py --model best_model.keras --images leaf1.jpg leaf2.jpg
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

CLASS_NAMES = [
    "Early Blight",
    "Healthy",
    "Late Blight",
]


def load_image_rgb(path, size=(256, 256)):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.resize(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA
    )


def last_conv_name(model):
    return [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]


def compute_gradcam(model, img_rgb, layer_name, class_index=None):
    x = np.expand_dims(img_rgb.astype(np.float32) / 255.0, 0)
    model(x)
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.squeeze(
        tf.reduce_sum(conv_out * tf.reshape(weights, [1, 1, 1, -1]), axis=-1)
    )
    cam = tf.maximum(cam, 0)
    cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    cam = cv2.resize(
        cam, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    return cam, class_index, float(tf.reduce_max(preds[0]))


def overlay(img_rgb, cam, alpha=0.5):
    heat = (plt.get_cmap("jet")(cam)[..., :3] * 255).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * img_rgb).clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument(
        "--layer", default=None, help="conv layer name (default: last Conv2D)"
    )
    a = ap.parse_args()
    model = tf.keras.models.load_model(a.model)
    layer = a.layer or last_conv_name(model)
    for i, p in enumerate(a.images):
        img = load_image_rgb(p)
        cam, idx, conf = compute_gradcam(model, img, layer)
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay(img, cam))
        plt.axis("off")
        plt.title(f"Grad-CAM: {CLASS_NAMES[idx]} ({conf:.2f})")
        plt.savefig(f"gradcam_{i + 1}.png", dpi=200, bbox_inches="tight")
