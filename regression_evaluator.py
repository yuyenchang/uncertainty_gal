# regression_evaluator.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics

class RegressionEvaluator:
    """Class to perform cross-validation training and evaluation.
    """

    def __init__(self, model, n_splits=3):
        """Initialize the evaluator with the model and number of splits.
        
        Parameters:
        - model: Model to evaluate.
        - n_splits (int): Number of folds for K-Fold cross-validation.
        """
        self.model = model  # Model to evaluate
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Set up K-Fold cross-validation

    def evaluate(self, X, y):
        """Perform cross-validation and evaluate model performance with uncertainty.
        
        Evaluates the model using Mean Squared Error (MSE) and 
        Mean Absolute Error (MAE) metrics. Captures prediction variability 
        to provide a measure of prediction confidence.
        
        Parameters:
        - X (numpy.ndarray): Input features for evaluation.
        - y (numpy.ndarray): True target values for evaluation.

        Returns:
        - y_pred_mean (numpy.ndarray): Array of predicted values for each instance.
        - y_pred_std (numpy.ndarray): Array of uncertainty (standard deviation) for each prediction.
        """
        mse_scores, mae_scores = [], []  # Lists to store evaluation metrics
        y_pred_mean, y_pred_std = np.zeros_like(y), np.zeros_like(y)  # Arrays to store predictions and uncertainties

        for train_idx, val_idx in self.kf.split(X):  # Perform K-Fold cross-validation
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.model.fit_model(X_train, y_train, X_val, y_val)  # Train model

            # Predict with uncertainty estimation method (e.g., dropout, bnn, ensemble, and bootstrap)
            y_pred, y_uncertainty = self.model.predict_with_uncertainty(X_val)
            y_pred_mean[val_idx], y_pred_std[val_idx] = y_pred, y_uncertainty  # Store predictions and uncertainties

            # Calculate MSE and MAE scores for this fold
            mse_scores.append(metrics.mean_squared_error(y_val, y_pred))  # Append MSE score
            mae_scores.append(metrics.mean_absolute_error(y_val, y_pred))  # Append MAE score

        # Calculate median and confidence intervals for metrics
        median_mse, median_mae = np.median(mse_scores), np.median(mae_scores)  # Median MSE and MAE
        # Compute 68% confidence intervals (16th and 84th percentiles) for MSE and MAE
        conf_interval_mse = [np.percentile(mse_scores, q) + np.mean(y_pred_std**2) for q in [16, 84]]
        conf_interval_mae = [np.percentile(mae_scores, q) + np.mean(y_pred_std) for q in [16, 84]]

        # Print metrics with confidence intervals and mean uncertainty
        print(
            f"MSE: {median_mse:.2f} +{abs(conf_interval_mse[1] - median_mse):.2f} -{abs(median_mse - conf_interval_mse[0]):.2f} | "
            f"MAE: {median_mae:.2f} +{abs(conf_interval_mae[1] - median_mae):.2f} -{abs(median_mae - conf_interval_mae[0]):.2f} | "
            f"Mean Uncertainty: {np.mean(y_pred_std):.2f}"
        )
        
        return y_pred_mean, y_pred_std  # Return predictions and uncertainties
