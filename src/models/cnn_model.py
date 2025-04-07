from tensorflow.keras import models, layers

# CNN Model for Hyperspectral Data
def HyperspectralCNN(in_channels, n_classes):
    """
    TensorFlow implementation of the HyperspectralCNN model.

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