# main.py

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessor import DataPreprocessor
from uncertainty_model import UncertaintyModel
from regression_evaluator import RegressionEvaluator

# Function to load the data
def load_data():
    """Load the tabular data and associated image files.
    
    This function integrates tabular and image data, 
    which allows for joint analysis of structured and unstructured data. 
    The use of tabular metadata with image data can improve model accuracy, 
    especially when predicting astrophysical targets that benefit from both types of features.
    """
    # Load table data from text file
    # The table file '_table.txt' contains target values and metadata for each data instance
    data = pd.read_csv('_table.txt', sep='\s+')
    
    # Extract target values and instance names
    name = data['name'].values  # Array of instance names
    y_16, y_true, y_84 = data['mstar16'].values, data['mstar50'].values, data['mstar84'].values  # Percentile values for target variables
    
    # Define the path where images are stored
    path = '_img/'
    
    # Initialize a 4D array to store images with shape (number of instances, channels, height, width)
    d0, d1, d2, d3 = len(name), 4, 33, 33
    images = np.full((d0, d1, d2, d3), -9999, dtype=float)  # -9999 used as placeholder
    
    # Initialize preprocessor to load and process images
    preprocessor = DataPreprocessor()
    
    # Load each image and assign to corresponding channels in the 4D array
    for i, idd in enumerate(name):
        idd = str(name[i])  # Convert name to string if not already
        # Load images for different channels and store them in the images array
        images[i, 0, :, :] = preprocessor.load_image(path + idd + '_JWST_f115w.fits')  # Channel 1
        images[i, 1, :, :] = preprocessor.load_image(path + idd + '_JWST_f150w.fits')  # Channel 2
        images[i, 2, :, :] = preprocessor.load_image(path + idd + '_JWST_f277w.fits')  # Channel 3
        images[i, 3, :, :] = preprocessor.load_image(path + idd + '_JWST_f444w.fits')  # Channel 4
    
    return images, y_16, y_true, y_84  # Return images and target values

# Function to preprocess data for modeling
def preprocess_data(images, y_true):
    """Normalize images and rescale target values for consistent model input.
    
    Normalization ensures that all data channels are on a comparable scale, 
    which can improve training stability and model performance.
    Target value rescaling to [0, 1] helps models with better convergence.
    """
    # Initialize preprocessor for image normalization
    preprocessor = DataPreprocessor()
    
    # Normalize each channel in the images using preprocessor
    X = preprocessor.normalize_channels(images)
    
    # Find min and max of target values for normalization
    y_min, y_max = y_true.min(), y_true.max()
    
    # Normalize target values between 0 and 1 for better model training
    y_normalized = (y_true - y_min) / (y_max - y_min)
    
    return X, y_normalized, y_min, y_max  # Return normalized images, target values, and original min/max values

# Function to evaluate different models with uncertainty estimation methods
def evaluate_models(X, y_normalized, y_min, y_max, y_true, y_16, y_84):
    """Evaluate models using various uncertainty estimation techniques and collect results.
    
    The choice of uncertainty estimation techniques ('dropout', 'bnn', 'ensemble', 'bootstrap') 
    provides flexibility in the analysis. Each method has distinct assumptions and strengths:
    
    - Dropout: Estimates uncertainty by randomly dropping units, useful for model uncertainty estimation.
    - BNN (Bayesian Neural Networks): Integrates Bayesian uncertainty, although computationally expensive.
    - Ensemble: Combines predictions from multiple models, reducing uncertainty and improving robustness.
    - Bootstrap: Resamples the dataset, effective for estimating statistical uncertainty.
    
    These techniques enable a better understanding of model predictions' reliability and robustness.
    """
    # Define model types to evaluate
    model_types = ["dropout", "bnn", "ensemble", "bootstrap"]
    
    # Dictionary to store results for each model type
    results = {}

    # Iterate over each model type to evaluate
    for model_type in model_types:
        print(f"Evaluating model with uncertainty method: {model_type}")
        
        # Initialize model with specified uncertainty type
        model = UncertaintyModel(input_shape=(4, 33, 33), model_type=model_type)
        
        # Initialize evaluator for regression tasks
        evaluator = RegressionEvaluator(model)
        
        # Evaluate model and obtain predictions and uncertainty estimates
        y_pred, y_uncertainty = evaluator.evaluate(X, y_normalized)
        
        # Rescale predictions and uncertainties back to original target value scale
        y_pred = y_pred * (y_max - y_min) + y_min
        y_uncertainty = y_uncertainty * (y_max - y_min)
        
        # Store predictions and uncertainties for this model type
        results[model_type] = (y_pred, y_uncertainty)

    return results  # Return results for all models

# Function to visualize uncertainty results
def plot_uncertainty_results(y_true, y_16, y_84, results, y_min, y_max):
    """Plot true vs predicted values with uncertainty bars for each model type.
    
    Visualization of model uncertainty provides insights into prediction confidence. 
    The use of error bars and perfect prediction lines helps validate model performance and 
    highlights discrepancies. This is essential for assessing if model uncertainty estimates 
    align with actual data variability.
    """
    # Set font size for all plots
    plt.rcParams.update({'font.size': 16})
    
    # Create subplots grid for each model type
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Original vs Predicted Values with Different Uncertainty Methods')

    # Iterate over each model type and plot results
    for i, (model_type, (y_pred, y_uncertainty)) in enumerate(results.items()):
        # Select subplot for this model
        ax = axs[i // 2, i % 2]

        # Scatter plot of original vs predicted values
        ax.scatter(y_true, y_pred, marker='.', s=3, alpha=0.8)
        
        # Add error bars for uncertainty around predicted values
        ax.errorbar(y_true, y_pred, xerr=[y_true - y_16, y_84 - y_true], 
                    yerr=y_uncertainty, fmt='.', ms=1, linewidth=0.5, alpha=0.3, label='Predicted with Uncertainty')
        
        # Add reference line for perfect predictions
        ax.plot([y_min, y_max], [y_min, y_max], color='red', linestyle='--', label='Perfect Prediction')
        
        # Set axis limits
        ax.set_xlim([y_min, y_max])
        ax.set_ylim([y_min, y_max])
        
        # Label axes and add title
        ax.set_xlabel('Original Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_type.upper()} Uncertainty')
        ax.legend()

    # Adjust layout and save plot to file
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('uncertainty_gal.png', format='png', dpi=300)
    plt.close()

# Main function to execute the steps
def main():
    """
    Main function to load data, preprocess it, evaluate models with uncertainty estimation, 
    and visualize the results.
    
    This workflow integrates data loading, preprocessing, model evaluation, and visualization.
    Each step contributes to a reliable uncertainty quantification framework for the model's predictions.
    """
    # Load data (images and target values)
    images, y_16, y_true, y_84 = load_data()
    
    # Preprocess images and normalize target values
    X, y_normalized, y_min, y_max = preprocess_data(images, y_true)
    
    # Evaluate models and collect predictions and uncertainties
    results = evaluate_models(X, y_normalized, y_min, y_max, y_true, y_16, y_84)
    
    # Plot and save results showing predictions with uncertainty for each model
    plot_uncertainty_results(y_true, y_16, y_84, results, y_min, y_max)

# Run the main function if the script is executed
if __name__ == '__main__':
    main()
