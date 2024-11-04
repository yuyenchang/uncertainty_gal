# Uncertainty Quantification for Galaxy Properties

This project analyzes James Webb Space Telescope (JWST) images using deep learning models to estimate stellar masses of galaxies and quantify uncertainty. The code includes data preprocessing, model training with various uncertainty estimation methods, and visualization of results.

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

(The code was written in Python 3.9.1.)

## Usage
1. **Prepare Data**:

Place your FITS image files in an _img/ directory.

Ensure _table.txt contains the necessary metadata and target values.

2. **Run the Main Script**:

 ```bash
python3 main.py
```

This will process the images, train models with different uncertainty estimation methods, and save the result plots in the directory.

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
└── uncertainty_gal.png         # Plot showing predicted vs. original values with uncertainty
 ```

## Uncertainty Estimation Methods
This project includes the following methods for uncertainty estimation:

1. **Dropout**: Uses Monte Carlo dropout to drop neurons and average forward passes for uncertainty.
2. **BNN**: Uses Bayesian layers with averaged forward passes to capture uncertainty.
3. **Ensemble**: Trains multiple models to estimate prediction uncertainty.
4. **Bootstrap**: Employs resampling techniques to estimate prediction uncertainty.

## Results

The project generates scatter plots of original vs. predicted galaxy properties (stellar masses) with uncertainty intervals for each method, revealing model accuracy and identifying the most reliable methods.

<img src="https://github.com/yuyenchang/uncertainty_gal/blob/main/uncertainty_gal.png" alt="Example Image" style="width:80%;"/>

## License
This project is licensed under the MIT License.
