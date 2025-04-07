import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, 
                                     ReLU, GlobalAveragePooling2D, Dense)
from tensorflow.keras import Model


class HyperspectralAE(tf.keras.Model):
    def __init__(self, in_channels, n_classes):
        super(HyperspectralAE, self).__init__()

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
        ])

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
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        classified = self.classifier(encoded)
        return decoded, classified