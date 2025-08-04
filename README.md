# Hyperspectral Image Classification (HIC)

## 🌐 Project Overview

Hyperspectral Image Classification (HIC) is an advanced machine learning project designed to extract meaningful insights from hyperspectral imagery using state-of-the-art deep learning techniques. This workflow combines Convolutional Neural Networks (CNN) and Autoencoder (AE) models to provide robust and accurate classification of hyperspectral data.

## 🔬 Scientific Background

Hyperspectral imaging captures electromagnetic radiation across numerous contiguous spectral bands, providing rich spectral information beyond traditional RGB imaging. This project addresses critical challenges in:
- Land use classification
- Agricultural monitoring
- Environmental research
- Remote sensing applications

## 🏗️ Project Architecture

### Key Components
1. **Data Processing**
   - **Data Ingestion**: Sophisticated loading and preprocessing of complex hyperspectral datasets
   - **Data Transformation**: Advanced feature extraction and normalization techniques

2. **Machine Learning Models**
   - **Convolutional Neural Network (CNN)**: 
     - Specialized architecture for spatial-spectral feature learning
     - High-performance classification across multiple spectral domains
   - **Autoencoder (AE)**:
     - Dimensionality reduction
     - Feature representation learning
     - Noise reduction and data compression

3. **Computational Pipelines**
   - **Training Pipeline**: End-to-end model training workflow
   - **Prediction Pipeline**: Seamless inference and classification

## 📂 Project Structure

```
HIC/
│
├── DATA/                   # Raw hyperspectral image datasets
│   ├── Botswana/           # Regional hyperspectral dataset
│   │   ├── Botswana.mat      # Hyperspectral image data
│   │   └── Botswana_gt.mat   # Ground truth labels
│   └── KSC/                # Kennedy Space Center dataset
│       ├── KSC.mat           # Hyperspectral image data
│       └── KSC_gt.mat        # Ground truth labels
│
├── artifacts/              # Project output
│   ├── trained_models/         
│   ├── visualizations/        
│   └── results/
│
├── config/                 # Configuration management
│   └── config.yaml            # Project configuration file
│
├── logs/                   # Execution and debugging logs
│
└── src/                    # Source code
    ├── __init__.py
    ├── exception.py           # Custom exception handling
    ├── logger.py             # Logging configuration
    ├── utils.py              # Utility functions
    │
    ├── components/         # Data processing modules
    │   ├── __init__.py
    │   ├── data_ingestion.py       # Data loading and preprocessing
    │   ├── data_transformation.py  # Feature extraction and normalization
    │   └── model_trainer.py        # Model training utilities
    │
    ├── models/             # Neural network architectures
    │   ├── __init__.py
    │   ├── ae_model.py            # Autoencoder model definition
    │   └── cnn_model.py           # CNN model definition
    │
    └── pipeline/           # Machine learning workflows
        ├── __init__.py
        ├── predict_pipeline.py      # Inference and prediction workflow
        └── train_pipeline.py        # Model training workflow
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Machine learning libraries (NumPy, Pandas, Scikit-learn)
- Deep learning framework (TensorFlow/PyTorch)
- CUDA-compatible GPU (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/HyperSpectral-AI.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Train the model
python -m src.pipeline.train_pipeline

# Run predictions
python -m src.pipeline.predict_pipeline
```

## 📊 Supported Datasets
- **Botswana Hyperspectral Dataset**
  - Download Dataset: [Botswana]('http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat')
  - Download Ground Truth: [Botswana_gt]('http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat')
  - Geographic region: Southern Africa
  - Spectral characteristics: High-resolution land cover classification
- **Kennedy Space Center (KSC) Hyperspectral Dataset**
  - Download Dataset: [KSC]('http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat')
  - Download Ground Truth: [KSC_gt]('http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat')
  - Geographic region: Florida, United States
  - Applications: Coastal and environmental monitoring
- **Pavia University (PaviaU) Dataset**
  - Download Dataset: [PaviaU]('http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat')
  - Download Ground Truth: [PaviaU_gt]('http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat')
  - Geographic region: Northern Italy
  - Applications: Urban area classification
- **Pavia Centre (PaviaC) Dataset**
  - Download Dataset: [PaviaC]('http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat')
  - Download Ground Truth: [PaviaC_gt]('http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat')
  - Geographic region: Pavia city center, Italy
  - Applications: Urban material and structure identification
- **Indian Pines Dataset**
  - Download Dataset: [IndianPines]('http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat')
  - Download Ground Truth: [IndianPines_gt]('http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat')
  - Geographic region: Northwestern Indiana, United States
  - Applications: Agricultural area classification

💡 **Pro Tip**: Download the datasets and them to the `DATA/` directory

## 🛠️ Development Principles
- Object-oriented design
- Modular architecture
- PEP 8 and PEP 257 compliance
- Comprehensive error handling
- Detailed logging

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

MIT License


