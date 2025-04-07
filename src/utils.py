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