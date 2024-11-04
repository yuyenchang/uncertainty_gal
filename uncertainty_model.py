# uncertainty_model.py

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from tensorflow_probability import layers as tfpl
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.metrics import MeanSquaredError, MeanAbsoluteError
from keras.preprocessing.image import ImageDataGenerator

class UncertaintyModel:
    """Class to define and train the Keras model with uncertainty estimation.
    
    This class provides multiple approaches for uncertainty estimation:
    - Dropout (Monte Carlo Dropout)
    - Bayesian Neural Networks (BNN)
    - Ensemble models
    - Bootstrap sampling
    These methods allow for both epistemic and aleatoric uncertainty quantification, 
    making predictions more reliable for downstream analysis.
    """

    def __init__(self, input_shape, model_type="dropout", n_ensemble=5, n_bootstrap=5):
        """
        Initialize the model with specified model type.
        
        Parameters:
        - input_shape (tuple): Shape of the input data.
        - model_type (str): Type of model to create, "dropout", "bnn", "ensemble", or "bootstrap".
        - n_ensemble (int): Number of models in the ensemble if model_type="ensemble" or "bootstrap".
        - n_bootstrap (int): Number of bootstrap samples if model_type="bootstrap".
        
        The chosen model type impacts the uncertainty quantification approach. 
        For example, ensembles and bootstraps provide diverse predictions, 
        while dropout approximates Bayesian inference.
        """
        self.model_type = model_type  # Store the type of model for uncertainty
        self.n_ensemble = n_ensemble if model_type == "ensemble" else n_bootstrap  # Set number of models if ensemble/bootstrap
        self.input_shape = input_shape  # Define input shape for model creation
        self.models = self.create_model(input_shape)  # Create the model or ensemble based on specified type

    def create_model(self, input_shape):
        """Create a model or ensemble of models based on specified type."""
        if self.model_type == "bnn":  # If model type is Bayesian Neural Network (BNN)
            return [self.create_bnn_model(input_shape)]  # Return list with a single BNN model
        elif self.model_type == "dropout":  # If model type is dropout (Monte Carlo Dropout)
            return [self.create_dropout_model(input_shape)]  # Return list with a single dropout model
        elif self.model_type in ["ensemble", "bootstrap"]:  # If model type is ensemble or bootstrap
            return [self.create_dropout_model(input_shape) for _ in range(self.n_ensemble)]  # Create multiple models

    def create_dropout_model(self, input_shape):
        """Create and compile a dropout model for uncertainty estimation.
        
        This architecture employs Conv2D layers with ReLU activations to learn 
        spatial patterns. Dropout layers (with `model.p` as retention probability) 
        enable Monte Carlo sampling during prediction, yielding uncertainty estimates.
        Mean Squared Error (MSE) is used as the loss function for regression tasks.
        """
        model = Sequential()  # Initialize a Sequential model
        model.p = 0.8  # Set retention probability for dropout
        model.weight_decay = 1e-8  # Set weight decay for regularization
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))  # Add Conv2D layer
        model.add(MaxPooling2D(pool_size=(2, 2)))  # Add MaxPooling2D layer to downsample
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # Add another Conv2D layer
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # Downsample with MaxPooling2D
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # Add a deeper Conv2D layer
        model.add(GlobalAveragePooling2D())  # Add Global Average Pooling layer for spatial dimension reduction
        model.add(Flatten())  # Flatten the output
        model.add(Dense(256, activation='relu'))   # Add Dense layer with 256 units
        model.add(Dropout(1-model.p))  # Add Dropout layer for regularization
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(model.weight_decay)))   # Add Dense layer with 128 units with L2 regularization
        model.add(Dropout(1-model.p))  # Add another Dropout layer
        model.add(Dense(64, activation='relu'))  # Add Dense layer with 64 units
        model.add(Dense(1, activation='linear'))  # Output layer with single output for regression
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[MeanSquaredError(), MeanAbsoluteError()])  # Compile model with MSE loss
        return model  # Return the compiled model

    def create_bnn_model(self, input_shape):
        """Create and compile a Bayesian neural network model with modified Dense layers.
        
        DenseFlipout layers are used to incorporate Bayesian inference directly 
        into the model, allowing for epistemic uncertainty estimation. While 
        computationally expensive, this approach improves confidence estimates 
        when data is limited or noisy.
        """
        model = Sequential()
        model.p = 1.0  # Set retention probability for dropout
        model.weight_decay = 0.0  # Set weight decay for regularization
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(tfpl.DenseFlipout(256, activation='relu'))  # DenseFlipout layer for Bayesian uncertainty
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(model.weight_decay)))  # Add L2 regularization
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[MeanSquaredError(), MeanAbsoluteError()])
        return model

    def predict_with_uncertainty(self, x, n_iter=100):
        """Predict with the specified model type to estimate uncertainty.
        
        If using dropout or BNN, Monte Carlo sampling is applied to calculate 
        prediction variance. Tau is calculated to adjust variance, considering 
        dropout probability and weight decay, providing an additional measure 
        of prediction confidence.
        """
        if self.model_type == "ensemble":
            return self.predict_with_ensemble(x)  # Use ensemble method
        elif self.model_type == "bootstrap":
            return self.predict_with_bootstrap(x)  # Use bootstrap method
        else:
            # Dropout or BNN using Monte Carlo sampling
            predictions = np.array([self.models[0](x, training=True).numpy() for _ in range(n_iter)])  # Get predictions with dropout
            mean_prediction = predictions.mean(axis=0).flatten()  # Calculate mean prediction
            variance = predictions.var(axis=0).flatten()  # Calculate variance
            # Calculate tau
            l = 1.0  # Set the value of l 
            N = predictions.shape[0]  # Use the number of predictions as the sample size
            epsilon = 1e-8  # Small constant to prevent division by zero
            tau = l**2 * (1 - self.models[0].p) / (2 * N * (self.models[0].weight_decay + epsilon))  # Compute tau with epsilon
            variance += (tau + 1./epsilon) **-1  # Adjust variance with tau
            return mean_prediction, np.sqrt(variance)  # Return mean prediction and adjusted uncertainty

    def predict_with_ensemble(self, x):
        """Predict with ensemble of models to estimate uncertainty.
        
        The ensemble approach leverages predictions from multiple models, 
        reducing variance and improving robustness. The spread of predictions 
        provides a natural measure of uncertainty, reflecting the model's 
        stability across different initializations.
        """
        ensemble_predictions = np.array([model.predict(x, verbose=0).flatten() for model in self.models])  # Predict with each ensemble model
        mean_prediction = ensemble_predictions.mean(axis=0)  # Average predictions
        uncertainty = ensemble_predictions.std(axis=0)  # Compute standard deviation for uncertainty
        return mean_prediction, uncertainty

    def predict_with_bootstrap(self, x):
        """Predict with bootstrap models to estimate uncertainty.
        
        Bootstrap sampling creates multiple training sets by sampling with replacement, 
        capturing the variability in the data. This method is effective for estimating 
        uncertainty in datasets that may be noisy or contain outliers.
        """
        bootstrap_predictions = np.array([model.predict(x, verbose=0).flatten() for model in self.models])  # Predict with each bootstrap sample
        mean_prediction = bootstrap_predictions.mean(axis=0)  # Average bootstrap predictions
        uncertainty = bootstrap_predictions.std(axis=0)  # Compute standard deviation for uncertainty
        return mean_prediction, uncertainty

    def fit_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the model, ensemble, or bootstrap models with early stopping and learning rate reduction.
        
        Data augmentation is applied to improve model generalization, particularly 
        important for image data. Early stopping prevents overfitting by stopping 
        training if validation performance plateaus, while learning rate reduction 
        ensures smoother convergence.
        """
        datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, fill_mode='nearest', data_format='channels_first')
        datagen.fit(X_train)  # Fit data generator to training data for data augmentation
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Early stopping to prevent overfitting
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)  # Reduce learning rate on plateau

        if self.model_type == "ensemble":
            for model in self.models:  # Train each model in the ensemble
                model.fit(datagen.flow(X_train, y_train, batch_size=32),
                          validation_data=(X_val, y_val),
                          epochs=epochs,
                          callbacks=[early_stopping, reduce_lr],
                          verbose=0)
        elif self.model_type == "bootstrap":
            for model in self.models:  # Train each model in bootstrap with sample
                idxs = np.random.choice(len(X_train), len(X_train), replace=True)  # Random indices for bootstrap sampling
                X_train_boot, y_train_boot = X_train[idxs], y_train[idxs]  # Create bootstrap sample
                model.fit(datagen.flow(X_train_boot, y_train_boot, batch_size=32),
                          validation_data=(X_val, y_val),
                          epochs=epochs,
                          callbacks=[early_stopping, reduce_lr],
                          verbose=0)
        else:
            self.models[0].fit(datagen.flow(X_train, y_train, batch_size=32),
                               validation_data=(X_val, y_val),
                               epochs=epochs,
                               callbacks=[early_stopping, reduce_lr],
                               verbose=0)
