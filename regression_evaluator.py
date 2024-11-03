# regression_evaluator.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics

class RegressionEvaluator:
    """Class to perform cross-validation training and evaluation."""

    def __init__(self, model, n_splits=3):
        self.model = model  # Model to evaluate
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Set up K-Fold cross-validation

    def evaluate(self, X, y):
        """Perform cross-validation and evaluate model performance with uncertainty."""
        mse_scores, mae_scores = [], []  # Lists to store evaluation metrics
        y_pred_mean, y_pred_std = np.zeros_like(y), np.zeros_like(y)  # Arrays to store predictions and uncertainties

        for train_idx, val_idx in self.kf.split(X):  # Perform K-Fold cross-validation
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.model.fit_model(X_train, y_train, X_val, y_val)  # Train model

            y_pred, y_uncertainty = self.model.predict_with_uncertainty(X_val)  # Predict with uncertainty estimation
            y_pred_mean[val_idx], y_pred_std[val_idx] = y_pred, y_uncertainty  # Store predictions and uncertainties

            mse_scores.append(metrics.mean_squared_error(y_val, y_pred))  # Append MSE score
            mae_scores.append(metrics.mean_absolute_error(y_val, y_pred))  # Append MAE score

        # Calculate confidence intervals
        median_mse, median_mae = np.median(mse_scores), np.median(mae_scores)  # Median MSE and MAE
        conf_interval_mse = [np.percentile(mse_scores, q) + np.mean(y_pred_std**2) for q in [16, 84]]  # MSE confidence intervals
        conf_interval_mae = [np.percentile(mae_scores, q) + np.mean(y_pred_std) for q in [16, 84]]  # MAE confidence intervals

        print(
            f"MSE: {median_mse:.2f} +{abs(conf_interval_mse[1] - median_mse):.2f} -{abs(median_mse - conf_interval_mse[0]):.2f} | "
            f"MAE: {median_mae:.2f} +{abs(conf_interval_mae[1] - median_mae):.2f} -{abs(median_mae - conf_interval_mae[0]):.2f} | "
            f"Mean Uncertainty: {np.mean(y_pred_std):.2f}"
        )
        
        return y_pred_mean, y_pred_std  # Return predictions and uncertainties

