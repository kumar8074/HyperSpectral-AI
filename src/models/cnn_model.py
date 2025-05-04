# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/models/cnn_model.py
# Description: This module defines the architecture of a CNN model for hyperspectral data.
#              It includes multiple convolutional layers, batch normalization, and fully connected layers.
# Author: LALAN KUMAR
# Created: [08-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

"""Defines the Convolutional Neural Network (CNN) model architecture.

Includes functions to build the Keras layers and compile the final model.
"""

import tensorflow as tf
from tensorflow.keras import models, layers

# CNN Model for Hyperspectral Data
def hyperspectral_cnn(in_channels, n_classes):
    """
    Builds the TensorFlow Keras layers for the Hyperspectral CNN model.

    Args:
        in_channels (int): Number of input channels (e.g., 145 for hyperspectral data).
        n_classes (int): Number of output classes.

    Returns:
        model: A TensorFlow Keras model.
    """
    model = models.Sequential()

    # First convolutional block
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(7, 7, in_channels)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # Second convolutional block
    model.add(layers.Conv2D(128, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # Third convolutional block
    model.add(layers.Conv2D(256, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # Fourth convolutional block
    model.add(layers.Conv2D(512, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # Fifth convolutional block
    model.add(layers.Conv2D(1024, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # Flatten the output tensor before passing it through the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    return model

def build_cnn_model(in_channels, n_classes, learning_rate):
    """Builds and compiles the Hyperspectral CNN model.

    Args:
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = hyperspectral_cnn(in_channels, n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model