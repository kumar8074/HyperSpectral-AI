# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/models/autoencoder_model.py
# Description: This file defines the architecture of a Autoencoder model for hyperspectral data.
#              It includes an encoder-decoder structure for feature extraction and a classifier for classification tasks.
# Author: LALAN KUMAR
# Created: [08-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

"""Defines the Autoencoder (AE) model architecture with a classifier head.

Includes a custom Keras model with encoder, decoder, and classifier components,
along with a function to compile the model with appropriate dual losses.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, 
                                     ReLU, GlobalAveragePooling2D, Dense)
from tensorflow.keras.metrics import MeanSquaredError, SparseCategoricalAccuracy
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import Model
import keras

class HyperspectralAE(tf.keras.Model):
    """Custom Keras Model combining an Autoencoder and a Classifier.

    The encoder compresses the input, the decoder reconstructs it, and the
    classifier predicts classes based on the encoded representation.

    Args:
        in_channels (int): Number of input spectral bands.
        n_classes (int): Number of output classes for the classifier.
    """
    def __init__(self, in_channels, n_classes, **kwargs):
        """Initializes the Encoder, Decoder, and Classifier components."""
        super(HyperspectralAE, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Encoder
        self.encoder = tf.keras.Sequential([
            Conv2D(64, 3, padding='same'),         # (7, 7) → (7, 7)
            BatchNormalization(),
            ReLU(),
            #MaxPool2D(pool_size=2, strides=1, padding='same'),  # (7, 7) → (7, 7)

            Conv2D(128, 3, padding='same'),        # (7, 7)
            BatchNormalization(),
            ReLU(),
            #MaxPool2D(pool_size=2, strides=2, padding='same'),  # (7, 7) → (4, 4)

            Conv2D(256, 3, padding='same'),        # (4, 4)
            BatchNormalization(),
            ReLU()
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            Conv2D(128, 3, padding='same'),  # (4, 4) → (8, 8)
            BatchNormalization(),
            ReLU(),

            Conv2D(64, 3, padding='same'),         # (8, 8) → (6, 6)
            BatchNormalization(),
            ReLU(),

            Conv2D(in_channels, 3, padding='same'),  # (6,6) → (7,7)
            BatchNormalization(),
            tf.keras.layers.Activation('sigmoid')
        ],name='decoder')

        # Classifier
        self.classifier = tf.keras.Sequential([
            Conv2D(512, 3, padding='same'),
            BatchNormalization(),
            ReLU(),

            Conv2D(1024, 3, padding='same'),
            BatchNormalization(),
            ReLU(),

            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dense(n_classes, activation='softmax')
        ],name='classifier')

    def call(self, inputs):
        """Forward pass defining the flow through encoder, decoder, and classifier.

        Args:
            inputs: The input tensor (batch of hyperspectral patches).

        Returns:
            A tuple containing (decoded_output, classification_output).
        """
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        return decoded, classification
    
    def get_config(self):
        """Returns the config of the model."""
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "n_classes": self.n_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a new instance from the config."""
        return cls(
            in_channels=config.pop("in_channels"),
            n_classes=config.pop("n_classes"),
            **config
        )

@keras.saving.register_keras_serializable(package="CustomModels")
def build_ae_model(in_channels, n_classes,learning_rate):
    """Builds and compiles the HyperspectralAE model.

    Compiles the model with Mean Squared Error for reconstruction loss and
    Sparse Categorical Crossentropy for classification loss.

    Args:
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = HyperspectralAE(in_channels, n_classes)
    recon_loss="mse"
    class_loss="sparse_categorical_crossentropy"
    recon_weight=0.5
    class_weight=0.5
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=[recon_loss, class_loss],
        loss_weights=[recon_weight, class_weight],
        metrics= [ MeanSquaredError(name="decoder_mse"), SparseCategoricalAccuracy(name="classifier_accuracy")]
    )

    return model
