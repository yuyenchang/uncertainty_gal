# data_preprocessor.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from astropy.io import fits

class DataPreprocessor:
    """Class for data loading and preprocessing functions."""

    def load_image(self, file_path):
        """Load image data from a FITS file.
        
        FITS format is commonly used in astrophysics for storing image data, 
        making it a suitable choice for loading and analyzing scientific images 
        in this model. The FITS file structure allows for metadata storage, 
        which can be useful for additional contextual information if needed.
        """
        with fits.open(file_path) as hdul:
            return hdul[0].data  # Opens FITS file and returns the image data from the first HDU (Header/Data Unit)

    def normalize_channels(self, images):
        """Normalize each channel in the image batch.
        
        Scaling images to a uniform range (0 to 1) allows for consistent model 
        training. Here, MinMaxScaler is used for scaling, as it is appropriate 
        for image data where pixel values typically range within known limits.
        Normalizing each channel individually can also help in cases where channels 
        might represent different physical properties, reducing bias toward any particular channel.
        """
        for i in range(images.shape[1]):  # Iterate over each channel in the image batch
            scaler = MinMaxScaler()  # Replace with StandardScaler() if desired
            reshaped = images[:, i, :, :].reshape(images.shape[0], -1)  # Flatten each image in this channel
            images[:, i, :, :] = scaler.fit_transform(reshaped).reshape(images.shape[0], images.shape[2], images.shape[3])  # Standardize each channel
        return images  # Return the normalized images

    def apply_pca(self, images, n_components=4):
        """Apply PCA to each image's channels for dimensionality reduction.
        
        PCA is used to reduce the dimensionality while preserving the main features 
        in the image data, especially beneficial for high-dimensional datasets like images. 
        This technique reduces computational cost and memory usage, making training faster 
        without significant information loss. Here, `n_components=4` is chosen to match 
        the original number of channels, maintaining interpretability across channels.
        """
        new_images = np.zeros((images.shape[0], n_components, images.shape[2], images.shape[3]))  # Initialize new array for transformed images
        for i in range(images.shape[0]):  # Iterate over each image in the batch
            reshaped_image = images[i].reshape(images.shape[1], -1).T  # Flatten each channel in the image for PCA
            pca = PCA(n_components=n_components)  # Initialize PCA with specified number of components
            transformed_image = pca.fit_transform(reshaped_image)  # Perform PCA transformation
            new_images[i] = transformed_image.T.reshape(n_components, images.shape[2], images.shape[3])  # Reshape and store the transformed image
        return new_images  # Return the PCA-reduced images


class UncertaintyModel:
    """Class to define and train the Keras model with uncertainty estimation."""

    def __init__(self, input_shape, model_type="dropout", n_ensemble=3, n_bootstrap=3):
        """
        Initialize the model with specified model type.
        
        Parameters:
        - input_shape (tuple): Shape of the input data.
        - model_type (str): Type of model to create, "dropout", "bnn", "ensemble", or "bootstrap".
        - n_ensemble (int): Number of models in the ensemble if model_type="ensemble" or "bootstrap".
        - n_bootstrap (int): Number of bootstrap samples if model_type="bootstrap".
        
        The choice of model type affects uncertainty quantification:
        - "dropout" applies Monte Carlo Dropout for variance estimation.
        - "bnn" uses Bayesian Neural Network layers for epistemic uncertainty.
        - "ensemble" and "bootstrap" leverage multiple models to reduce bias and variance.
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
        
        The model uses Conv2D layers with ReLU activations to capture non-linear spatial features.
        Dropout layers are added to approximate Bayesian inference, using Monte Carlo Dropout 
        during prediction to estimate predictive uncertainty. The choice of `loss='mean_squared_error'` 
        for regression aligns with the goal of minimizing prediction error. Activation functions and 
        architecture are chosen to balance computational efficiency and model complexity.
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
        
        Bayesian layers (`DenseFlipout`) provide epistemic uncertainty, capturing 
        model uncertainty, which is valuable in cases where the dataset may not 
        cover all possible scenarios. `tfpl.DenseFlipout` layers are more computationally 
        demanding than standard dense layers but offer improved uncertainty estimates.
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
        
        For dropout and BNN, Monte Carlo sampling is used to obtain uncertainty estimates. 
        A large number of iterations (`n_iter=100`) is set to ensure stable uncertainty 
        quantification by capturing prediction variance. `tau` is calculated to account for 
        the effect of dropout probability and weight decay, providing a robust estimation of predictive uncertainty.
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
            tau = l**2 * (1 - self.models[0].p) / (2 * N * (self.models[0].weight_decay + epsilon))+ epsilon  # Compute tau with epsilon
            variance += tau**-1  # Adjust variance with tau
            return mean_prediction, np.sqrt(variance)  # Return mean prediction and adjusted uncertainty

    def predict_with_ensemble(self, x):
        """Predict with ensemble of models to estimate uncertainty.
        
        The ensemble method combines multiple models trained on the same data, 
        providing diverse perspectives on predictions. Variance across predictions 
        is used to estimate uncertainty, particularly effective for capturing 
        aleatoric uncertainty in the dataset.
        """
        ensemble_predictions = np.array([model.predict(x, verbose=0).flatten() for model in self.models])  # Predict with each ensemble model
        mean_prediction = ensemble_predictions.mean(axis=0)  # Average predictions
        uncertainty = ensemble_predictions.std(axis=0)  # Compute standard deviation for uncertainty
        return mean_prediction, uncertainty

    def predict_with_bootstrap(self, x):
        """Predict with bootstrap models to estimate uncertainty.
        
        Bootstrap replicates the training process with different data samples, 
        which helps in capturing dataset variability. The approach is robust in 
        estimating uncertainty where data may be noisy or contain outliers.
        """
        bootstrap_predictions = np.array([model.predict(x, verbose=0).flatten() for model in self.models])  # Predict with each bootstrap sample
        mean_prediction = bootstrap_predictions.mean(axis=0)  # Average bootstrap predictions
        uncertainty = bootstrap_predictions.std(axis=0)  # Compute standard deviation for uncertainty
        return mean_prediction, uncertainty

    def fit_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the model, ensemble, or bootstrap models with early stopping and learning rate reduction.
        
        Data augmentation (rotation, horizontal flip) is applied to enrich the training set, 
        making the model more robust to variations in input. Early stopping and learning rate 
        reduction help prevent overfitting by stopping training at an optimal point and adjusting 
        the learning rate for smooth convergence.
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
