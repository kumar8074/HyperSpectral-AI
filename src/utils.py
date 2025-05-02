# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/utils.py
# Description: This file contains the various utility functions used in the hyperspectral image classification project.
# Author: LALAN KUMAR
# Created: [07-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import yaml
import scipy.io as sio
import logging
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import sys

from .logger import logging as project_logger
from .exception import CustomException

################################ USED IN DATA INGESTION ################################

# Load a YAML file from the given file path
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

# Load hyperspectral data and labels from .mat files
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
    project_logger.debug(f"Keys in the image file '{image_file}': {image_data.keys()}")
    
    # Extract the image data using the specified key
    images = image_data.get(image_key)
    
    if images is None:
        raise KeyError(f"Image data not found in the key '{image_key}' in the file {image_file}.")
    
    # Load the label data
    label_data = sio.loadmat(os.path.join(dataset_path, label_file))
    project_logger.debug(f"Keys in the label file '{label_file}': {label_data.keys()}")
    
    # Extract the label data using the specified key
    labels = label_data.get(label_key)
    
    if labels is None:
        raise KeyError(f"Label data not found in the key '{label_key}' in the file {label_file}.")
    
    # Output shapes and return the data
    project_logger.debug(f"Loaded Image shape: {images.shape}")
    project_logger.debug(f"Loaded Label shape: {labels.shape}")
    project_logger.debug(f"Unique labels: {set(labels.flatten())}")

    return images, labels

################################ USED IN DATA INGESTION ############################################

################################ USED IN DATA TRANSFORMATION ########################################

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
def normalize_patches(patches, method='pca_output'): # For AE use per_band.
    """
    Normalize hyperspectral image patches or PCA-reduced patches.

    Parameters:
        patches (np.ndarray): Input array of shape (num_patches, height, width, channels/components).
        method (str): One of 'per_band', 'per_patch', 'pca_output'.

    Returns:
        np.ndarray: Normalized patches of the same shape.
    """
    patches = patches.astype(np.float32)

    if np.isnan(patches).any():
        raise ValueError("Input contains NaNs before normalization.")

    if method == 'per_band':
        # Suitable for AE+Classifier (original spectral bands)
        num_patches, height, width, channels = patches.shape
        reshaped = patches.reshape(-1, channels)

        # Clip extreme values to improve robustness
        reshaped = np.clip(
            reshaped,
            np.percentile(reshaped, 1, axis=0),
            np.percentile(reshaped, 99, axis=0)
        )

        min_vals = reshaped.min(axis=0)
        max_vals = reshaped.max(axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1  # Avoid division by zero

        normalized = (reshaped - min_vals) / denom
        return normalized.reshape(num_patches, height, width, channels)

    elif method == 'per_patch':
        # Normalize each patch individually (can be noisy)
        min_vals = patches.min(axis=(1, 2, 3), keepdims=True)
        max_vals = patches.max(axis=(1, 2, 3), keepdims=True)
        denom = max_vals - min_vals
        denom[denom == 0] = 1

        return (patches - min_vals) / denom

    elif method == 'pca_output':
        # Suitable after PCA (for CNN workflow)
        num_patches, height, width, components = patches.shape
        reshaped = patches.reshape(-1, components)

        min_vals = reshaped.min(axis=0)
        max_vals = reshaped.max(axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1

        normalized = (reshaped - min_vals) / denom
        return normalized.reshape(num_patches, height, width, components)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


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
    except Exception as e:
        raise CustomException(e,sys)

# Splits the data into training and testing sets
def preprocess_and_split(patches, labels, test_size=0.2, batch_size=32,ae=False):
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
    if patches.shape[0] != labels.shape[0]:
        raise ValueError("Mismatch between patches and labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels, test_size=test_size, stratify=labels, random_state=42
    )

    if ae:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (X_train, y_train)))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, (X_test, y_test)))
        logging.info("AutoEncoder Workflow TF datasets successfully created.")
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        logging.info("CNN Workflow TF datasets successfully created.")


    train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset, X_train, X_test, y_train, y_test

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

################################ USED IN DATA TRANSFORMATION ########################################

################################ USED IN MODEL TRAINING #############################################
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

################################ USED IN MODEL TRAINING #############################################
    





