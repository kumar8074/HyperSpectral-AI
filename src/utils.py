import os
import scipy.io as sio
import sys
import yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import logging
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint

from src.logger import logging  # Logging setup
from src.exception import CustomException  # Custom exception class

# Function to load the configuration from a YAML file
def load_yaml(file_path):
    """
    Load a YAML file from the given file path.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: The data loaded from the YAML file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found at: {file_path}")

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data


#################### UTILIZED IN DATA INGESTION ############################
# Function to load the Hyperspectral data
def load_hyperspectral_data(data_dir, dataset_name, config):
    """
    Load hyperspectral data and labels from .mat files.

    Args:
        data_dir (str): Directory containing the .mat files.
        dataset_name (str): The name of the dataset (e.g., 'Botswana', 'PaviaC', etc.).
        config (dict): The configuration dictionary from the config.yaml file.

    Returns:
        images (ndarray): Hyperspectral image data.
        labels (ndarray): Ground-truth labels.
    """
    # Retrieve dataset configuration from the YAML config
    dataset_config = next((dataset for dataset in config['datasets'] if dataset['name'] == dataset_name), None)
    
    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in the config.")

    dataset_path = os.path.join(data_dir, dataset_config['name'])
    image_pattern = dataset_config['image_pattern']
    label_pattern = dataset_config['label_pattern']
    image_key = dataset_config['image_key']
    label_key = dataset_config['label_key']

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder '{dataset_name}' not found in '{dataset_path}'.")

    # Identify image and label files in the dataset directory
    image_file = next((f for f in os.listdir(dataset_path) if f.endswith('.mat') and image_pattern in f), None)
    label_file = next((f for f in os.listdir(dataset_path) if f.endswith('.mat') and label_pattern in f), None)
    
    if not image_file or not label_file:
        raise FileNotFoundError(f"Image or label .mat files not found in the '{dataset_name}' dataset directory.")

    # Load the image data
    image_data = sio.loadmat(os.path.join(dataset_path, image_file))
    #print(f"Keys in the image file '{image_file}':", image_data.keys())
    
    # Extract the image data using the specified key
    images = image_data.get(image_key)
    
    if images is None:
        raise KeyError(f"Image data not found in the key '{image_key}' in the file {image_file}.")
    
    # Load the label data
    label_data = sio.loadmat(os.path.join(dataset_path, label_file))
    #print(f"Keys in the label file '{label_file}':", label_data.keys())
    
    # Extract the label data using the specified key
    labels = label_data.get(label_key)
    
    if labels is None:
        raise KeyError(f"Label data not found in the key '{label_key}' in the file {label_file}.")
    
    # Output shapes and return the data
    #print(f"Image shape: {images.shape}")
    #print(f"Label shape: {labels.shape}")
    #print(f"Unique labels: {set(labels.flatten())}")

    return images, labels
#################### UTILIZED IN DATA INGESTION ############################


#################### UTILIZED IN DATA TRANSFORMATION ############################
# Applies PCA to the hyperspectral data (only used for CNN Based Approach)
def apply_pca(images, n_components):
    """
    Apply PCA to reduce the dimensionality of the hyperspectral data.
    Args:
        images (ndarray): Hyperspectral image data of shape (H, W, C).
        n_components (int): Number of principal components to retain.
    Returns:
        reduced_images (ndarray): PCA-reduced hyperspectral image data of shape (H, W, n_components).
    """
    h, w, c = images.shape
    n_components = min(n_components, c)  # Ensure PCA components don't exceed available features
    reshaped_images = images.reshape(-1, c)  # Reshape to (H*W, C)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(reshaped_images)
    reduced_images = reduced_data.reshape(h, w, n_components)  # Reshape back to (H, W, n_components)
    return reduced_images


# Extracts patches from Hyperspectral data (Utilized both for CNN and AutoEncoder based Approaches.)
def extract_patches(images, labels=None, patch_size=7):
    """
    Extract patches from the hyperspectral image. Optionally align with valid labels if provided.

    Args:
        images (ndarray): Hyperspectral image data of shape (H, W, C).
        labels (ndarray, optional): Ground-truth labels of shape (H, W). Defaults to None.
        patch_size (int): Size of the patch to extract (e.g., 7 for 7x7 patches).

    Returns:
        patches (ndarray): Extracted patches of shape (num_patches, patch_size, patch_size, C).
        valid_labels (ndarray or None): The corresponding labels for each patch (if labels are provided).
    """
    patches = []
    valid_labels = []
    pad = patch_size // 2

    # Pad images with zeros for patch extraction
    padded_images = np.pad(images, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    # Pad labels if provided, otherwise set to None
    padded_labels = np.pad(labels, ((pad, pad), (pad, pad)), mode='constant') if labels is not None else None

    for i in range(pad, padded_images.shape[0] - pad):
        for j in range(pad, padded_images.shape[1] - pad):
            patch = padded_images[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            if labels is None or (padded_labels is not None and padded_labels[i, j] != 0):
                patches.append(patch)
                if labels is not None:
                    valid_labels.append(padded_labels[i, j])

    patches = np.array(patches)
    valid_labels = np.array(valid_labels) if labels is not None else None

    if patches.shape[0] == 0:
        raise ValueError("No valid patches found, check label distribution.")

    return patches, valid_labels


# Noramalizes the extracted patches
def normalize_patches(patches):
    patches = patches.astype(np.float32)
    num_patches, height, width, channels = patches.shape
    reshaped = patches.reshape(-1, channels)
    min_vals = reshaped.min(axis=0)
    max_vals = reshaped.max(axis=0)
    denom = max_vals - min_vals
    denom[denom == 0] = 1
    normalized = (reshaped - min_vals) / denom
    return normalized.reshape(num_patches, height, width, channels)

# Saves the transformer object
def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Args:
        file_path (str): Path where the object will be saved.
        obj: Object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        #logging.info(f"Object saved at {file_path}.")
    except Exception as e:
        #logging.error(f"Failed to save object at {file_path}: {e}")
        raise CustomException(e,sys)
    
    
    
# Splits the data into training and testing sets
def preprocess_and_split(patches, labels=None, test_size=0.2, batch_size=32):
    """
    Preprocess hyperspectral data and split into train/test sets.

    Args:
        patches (ndarray): Extracted patches of shape (num_patches, patch_size, patch_size, C).
        labels (ndarray, optional): Ground-truth labels (for supervised learning).
        test_size (float): Fraction of data to use for testing.
        batch_size (int): Batch size for TensorFlow datasets.

    Returns:
        train_dataset, test_dataset: TensorFlow datasets for training and testing.
    """
    if labels is not None:
        if patches.shape[0] != labels.shape[0]:
            raise ValueError("Mismatch between patches and labels.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            patches, labels, test_size=test_size, random_state=42, stratify=labels
        )
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        # For AutoEncoder: inputs = targets
        X_train, X_test = train_test_split(
            patches, test_size=test_size, random_state=42
        )
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)).shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


# Loads the transformer object
def load_transformer(transformer_path):
    """
    Load a saved transformer object from the specified path.

    Args:
        transformer_path (str): Path to the saved transformer object.

    Returns:
        dict: Loaded transformer object.

    Raises:
        CustomException: If any error occurs during loading.
    """
    try:
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Transformer file not found at path: {transformer_path}")
        
        with open(transformer_path, 'rb') as file:
            transformer = pickle.load(file)
        
        logging.info(f"Transformer object loaded from {transformer_path}.")
        return transformer
    except Exception as e:
        logging.error(f"Error occurred while loading transformer object: {e}")
        raise CustomException(e, sys)
#################### UTILIZED IN DATA TRANSFORMATION ############################


#################### UTILIZED IN MODEL TRAINER ##################################
def get_optimizer(optimizer_name):
    """
    Get the optimizer by name.
    
    Args:
        optimizer_name (str): The name of the optimizer to use.
        
    Returns:
        The TensorFlow optimizer class.
    """

    optimizer_dict = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'adagrad': tf.keras.optimizers.Adagrad,
        'adadelta': tf.keras.optimizers.Adadelta,
        'adamax': tf.keras.optimizers.Adamax,
        'nadam': tf.keras.optimizers.Nadam
    }
    
    if optimizer_name.lower() not in optimizer_dict:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported. Available options: {list(optimizer_dict.keys())}")
    
    return optimizer_dict[optimizer_name.lower()]


def get_loss(loss_name):
    """
    Get the loss function by name.
    
    Args:
        loss_name (str): The name of the loss function to use.
        
    Returns:
        The TensorFlow loss function.
    """
    
    loss_dict = {
        'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy,
        'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy,
        'binary_crossentropy': tf.keras.losses.BinaryCrossentropy,
        'mean_squared_error': tf.keras.losses.MeanSquaredError,
        'mean_absolute_error': tf.keras.losses.MeanAbsoluteError,
        'huber': tf.keras.losses.Huber,
        'kullback_leibler_divergence': tf.keras.losses.KLDivergence
    }
    
    if loss_name.lower() not in loss_dict:
        raise ValueError(f"Loss function '{loss_name}' not supported. Available options: {list(loss_dict.keys())}")
    
    return loss_dict[loss_name.lower()]()


def get_metric(metric_name):
    """
    Get the metric by name.
    
    Args:
        metric_name (str): The name of the metric to use.
        
    Returns:
        The TensorFlow metric object.
    """
    
    metric_dict = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy,
        'sparse_categorical_accuracy': tf.keras.metrics.SparseCategoricalAccuracy,
        'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy,
        'precision': tf.keras.metrics.Precision,
        'recall': tf.keras.metrics.Recall,
        'auc': tf.keras.metrics.AUC,
        'mae': tf.keras.metrics.MeanAbsoluteError,
        'mse': tf.keras.metrics.MeanSquaredError
    }
    
    if metric_name.lower() not in metric_dict:
        raise ValueError(f"Metric '{metric_name}' not supported. Available options: {list(metric_dict.keys())}")
    
    return metric_dict[metric_name.lower()]()


def train_model(model, train_dataset, test_dataset, save_model_path, epochs, callbacks=None):
    """
    Train a TensorFlow model.
    
    Args:
        model: The TensorFlow Keras model to train.
        train_dataset: Training dataset.
        test_dataset: Validation/testing dataset.
        save_model_path (str): Path to save the trained model.
        epochs (int): Number of epochs to train for.
        callbacks (list, optional): List of Keras callbacks to use during training.
        
    Returns:
        history: The training history.
    """
    
    # Create default callbacks if none provided
    if callbacks is None:
        callbacks = [
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # Model checkpoint to save the best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            # Reduce learning rate when a metric has stopped improving
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard for visualizing training progress
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(os.path.dirname(save_model_path), 'tensorboard_logs'),
                histogram_freq=1
            )
        ]
    
    # Train the model
    logging.info(f"Starting model training for {epochs} epochs")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model if it wasn't saved by callbacks
    if not any(isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks):
        model.save(save_model_path)
        logging.info(f"Model saved to {save_model_path}")
    
    return history
#################### UTILIZED IN MODEL TRAINER ##################################  


#################### UTILIZED IN MODEL EVALUATION ##################################
def evaluate_model(model, test_dataset, class_names=None):
    """
    Evaluate a trained model on a test dataset.
    
    Args:
        model: The trained TensorFlow Keras model.
        test_dataset: The test dataset.
        class_names (list, optional): List of class names for the confusion matrix.
        
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    
    # Evaluate the model
    logging.info("Evaluating model on test dataset")
    evaluation = model.evaluate(test_dataset, verbose=1)
    
    # Get model metric names
    metric_names = model.metrics_names
    
    # Create a dictionary of metrics
    metrics_dict = dict(zip(metric_names, evaluation))
    
    # If class names are provided, compute classification report and confusion matrix
    if class_names is not None:
        # Extract predictions and true labels
        y_pred = []
        y_true = []
        
        for images, labels in test_dataset:
            predictions = model.predict(images)
            if isinstance(predictions, list):  # For AutoEncoder, predictions is [decoded, classified]
                predictions = predictions[1]
                
            # Convert one-hot encoded or logits to class indices
            if predictions.shape[-1] > 1:  # Multi-class classification
                predictions = np.argmax(predictions, axis=-1)
            else:  # Binary classification
                predictions = (predictions > 0.5).astype(int)
            
            # If labels are one-hot encoded, convert to class indices
            if len(labels.shape) > 1 and labels.shape[-1] > 1:
                labels = np.argmax(labels, axis=-1)
            
            y_pred.extend(predictions)
            y_true.extend(labels.numpy())
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Add to metrics dictionary
        metrics_dict['confusion_matrix'] = cm
        metrics_dict['classification_report'] = report
    
    return metrics_dict  







