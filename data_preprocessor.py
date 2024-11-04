# data_preprocessor.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from astropy.io import fits

class DataPreprocessor:
    """Class for data loading and preprocessing functions.
    """

    def load_image(self, file_path):
        """Load image data from a FITS file.
        
        FITS format is commonly used in astrophysics for storing image data, 
        allowing for the inclusion of metadata useful for additional information.
        """
        with fits.open(file_path) as hdul:
            return hdul[0].data  # Opens FITS file and returns the image data from the first HDU (Header/Data Unit)

    def normalize_channels(self, images):
        """Normalize each channel in the image batch.
        
        This process ensures consistent pixel value scaling across channels, 
        reducing bias and enhancing model training effectiveness.
        """
        for i in range(images.shape[1]):  # Iterate over each channel in the image batch
            scaler = MinMaxScaler()  # Replace with StandardScaler() if desired
            reshaped = images[:, i, :, :].reshape(images.shape[0], -1)  # Flatten each image in this channel
            images[:, i, :, :] = scaler.fit_transform(reshaped).reshape(images.shape[0], images.shape[2], images.shape[3])  # Standardize each channel
        return images  # Return the normalized images

    def apply_pca(self, images, n_components=4):
        """Apply PCA to each image's channels for dimensionality reduction.
        
        PCA reduces dimensionality while preserving key features in high-dimensional datasets like images, 
        reducing computational cost and memory usage. 
        """
        new_images = np.zeros((images.shape[0], n_components, images.shape[2], images.shape[3]))  # Initialize new array for transformed images
        for i in range(images.shape[0]):  # Iterate over each image in the batch
            reshaped_image = images[i].reshape(images.shape[1], -1).T  # Flatten each channel in the image for PCA
            pca = PCA(n_components=n_components)  # Initialize PCA with specified number of components
            transformed_image = pca.fit_transform(reshaped_image)  # Perform PCA transformation
            new_images[i] = transformed_image.T.reshape(n_components, images.shape[2], images.shape[3])  # Reshape and store the transformed image
        return new_images  # Return the PCA-reduced images
