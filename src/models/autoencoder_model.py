# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/models/autoencoder_model.py
# Description: This module defines the architecture of a Autoencoder model for hyperspectral data.
#              It includes an encoder-decoder structure for feature extraction and a classifier for classification tasks.
# Author: LALAN KUMAR
# Created: [08-01-2025]
# Updated: [14-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, 
                                     ReLU, GlobalAveragePooling2D, Dense)
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import Model
import keras

@keras.saving.register_keras_serializable(package="CustomModels")
class HyperspectralAE(tf.keras.Model):
    def __init__(self, in_channels, n_classes, **kwargs):
        super(HyperspectralAE, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Encoder
        self.encoder = tf.keras.Sequential([
            Conv2D(64, 3, padding='same'),         # (7, 7) → (7, 7)
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=2, strides=1, padding='same'),  # (7, 7) → (7, 7)

            Conv2D(128, 3, padding='same'),        # (7, 7)
            BatchNormalization(),
            ReLU(),
            MaxPool2D(pool_size=2, strides=2, padding='same'),  # (7, 7) → (4, 4)

            Conv2D(256, 3, padding='same'),        # (4, 4)
            BatchNormalization(),
            ReLU()
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(128, 3, strides=2, padding='same'),  # (4, 4) → (8, 8)
            BatchNormalization(),
            ReLU(),

            Conv2D(64, 3, padding='valid'),         # (8, 8) → (6, 6)
            BatchNormalization(),
            ReLU(),

            Conv2DTranspose(in_channels, 2, strides=1, padding='valid'),  # (6,6) → (7,7)
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
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        return decoded, classification
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "n_classes": self.n_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels=config.pop("in_channels"),
            n_classes=config.pop("n_classes"),
            **config
        )