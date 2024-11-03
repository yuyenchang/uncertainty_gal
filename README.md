# Uncertainty Quantification for Galaxy Properties

This project analyzes James Webb Space Telescope (JWST) images using deep learning models to estimate galaxy properties and quantify uncertainty. The code includes data preprocessing, model training with various uncertainty estimation methods, and visualization of results.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Uncertainty Estimation Methods](#uncertainty-estimation-methods)
- [Results](#results)
- [License](#license)

## Project Overview

The Galaxy Project processes astronomical data, specifically galaxy images in FITS format, and applies deep learning techniques to predict specific properties of galaxies. This project aims to implement several uncertainty estimation techniques, allowing for more robust predictions with a quantified level of uncertainty.

## Features

- **Data Preprocessing**: Loads and normalizes galaxy images from FITS files.
- **Deep Learning Models**: Implements convolutional neural networks (CNNs) for regression tasks.
- **Uncertainty Estimation**: Supports various methods, including dropout, Bayesian neural networks, ensemble, and bootstrap.
- **Visualization**: Plots true vs. predicted values with uncertainty intervals.

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yuyenchang/uncertainty_gal.git
cd uncertainty_gal
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

Make sure requirements.txt includes necessary packages such as numpy, pandas, tensorflow, keras, astropy, scikit-learn, and scipy.

## Usage
1. **Prepare Data**:

Place your FITS image files in an _img/ directory.
Ensure _table.txt contains the necessary metadata and target values.

2. **Run the Main Script**:

 ```bash
python3 main.py
```

This will process the images, train models with different uncertainty estimation methods, and save the result plots in the  directory.

3. **Results**:

Generated plots show original vs. predicted values along with uncertainty intervals for each method.

## Directory Structure

 ```graphql
uncertainty_gal/
├── main.py                     # Main script to run the entire pipeline
├── data_preprocessor.py        # Handles data loading and preprocessing
├── uncertainty_model.py        # Defines the model architecture and uncertainty estimation
├── regression_evaluator.py     # Performs evaluation and cross-validation
├── _table.txt                  # Metadata and target values for images
├── _img/                       # Directory containing galaxy image files in FITS format
├── requirements.txt            # List of required packages
└── uncertainty_gal.png         # Plot showing predicted vs. actual values with uncertainty
 ```

## Uncertainty Estimation Methods
This project includes the following methods for uncertainty estimation:

1. **Dropout**: Monte Carlo Dropout to estimate uncertainty by randomly dropping neurons during inference.
2. **Bayesian Neural Network (BNN)**: Uses Bayesian layers (DenseFlipout) for uncertainty estimation.
3. **Ensemble**: Trains multiple models and computes the variance in predictions.
4. **Bootstrap**: Uses resampling techniques to estimate the uncertainty of predictions.

## Results
For each uncertainty estimation method, the project generates scatter plots of true vs. predicted galaxy properties with uncertainty intervals. These plots provide insights into model accuracy and uncertainty levels, helping to identify the most reliable methods for this dataset.

## License
This project is licensed under the MIT License.
