import os
import pickle
import numpy as np
import random
import torch

from sklearn.model_selection import train_test_split
from utils.folders import create_folder


class DataGenerator:
    """
    A class to generate synthetic datasets for machine learning models based on a specified configuration.
    The datasets include functions and related data for training and validation.
    """
    def __init__(self, config, seed=42):
        """
        Initializes the DataGenerator instance with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing settings for data generation.
            seed (int, optional): Seed for random number generation (default is 42).
        """
        self.PATH = config['PATH']  # Directory path to save generated data
        self.MODEL_NAME = config['MODEL_NAME']  # Name of the model
        self.N_DATA = config['N_DATA']  # Number of data points to generate
        self.N_DISCRETIZATION = config['N_DISCRETIZATION']  # Number of discretization points along axes
        self.x0, self.xN = config['x0'], config['xN']  # Range for x-axis
        self.y0, self.yN = config['y0'], config['yN']  # Range for y-axis
        self.custom_seed = seed  # Custom seed for reproducibility

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.x, self.x_step_size = self._create_axis(self.x0, self.xN)
        self.y, self.y_step_size = self._create_axis(self.y0, self.yN)
        self.X_mesh, self.Y_mesh = np.meshgrid(self.x, self.y)
        
        self.g1 = np.random.rand(self.N_DATA) 
        self.g2 = np.random.rand(self.N_DATA) 
        self.g3 = np.random.rand(self.N_DATA) 
    
    def _create_axis(self, start, end):
        """
        Creates a discretized axis between two points.

        Args:
            start (float): Start of the axis.
            end (float): End of the axis.

        Returns:
            tuple: A tuple containing the discretized axis and the step size.
        """
        axis = np.linspace(start, end, self.N_DISCRETIZATION).astype(np.float32)
        step_size = (end - start) / (self.N_DISCRETIZATION - 1) 
        return axis, step_size

    def _expand_dims(self, g1, g2, g3, X, Y):
        """
        Expands the dimensions of input arrays for broadcasting in calculations.

        Args:
            g1 (array): First generator output.
            g2 (array): Second generator output.
            X (array): Meshgrid X coordinates.
            Y (array): Meshgrid Y coordinates.

        Returns:
            tuple: Expanded versions of g1, g2, X, and Y.
        """
        g1 = g1[:, np.newaxis, np.newaxis] 
        g2 = g2[:, np.newaxis, np.newaxis] 
        g3 = g3[:, np.newaxis, np.newaxis]
        X = X[np.newaxis, :, :] 
        Y = Y[np.newaxis, :, :]  
        return g1, g2, g3, X, Y

    def u_func(self, g1, g2, g3, X, Y):
        """
        Calculates a function 'u' based on inputs g1, g2, and meshgrid coordinates.

        Args:
            g1 (array): First generator output.
            g2 (array): Second generator output.
            X (array): Meshgrid X coordinates.
            Y (array): Meshgrid Y coordinates.

        Returns:
            array: The computed 'u' values.
        """
        g1, g2, g3, X, Y = self._expand_dims(g1, g2, g3, X, Y)  # Expand dimensions for calculations
        return np.sqrt(g1 + g2*X + g3*Y)  # Calculate u based on g1 and g2

    def qx_func(self, g1, g2, g3, X, Y):
        """
        Computes the qx function based on inputs g1, g2 and g3.

        Args:
            g1 (array): First generator output.
            g2 (array): Second generator output.
            X (array): Meshgrid X coordinates.
            Y (array): Meshgrid Y coordinates.

        Returns:
            array: The computed qx values.
        """
        g1, g2, g3, X, Y = self._expand_dims(g1, g2, g3, X, Y)  # Expand dimensions for calculations
        return 1/2 * g2 * (1 - np.sqrt(g1 + g2*X + g3*Y))  # Calculate qx based on g1 and g2
        # return 1/2 * g2 * np.ones_like(X)


    def qy_func(self, g1, g2, g3, X, Y):
        """
        Computes the qy function based on inputs g1, g2 and g3.

        Args:
            g1 (array): First generator output.
            g2 (array): Second generator output.
            X (array): Meshgrid X coordinates.
            Y (array): Meshgrid Y coordinates.

        Returns:
            array: The computed qy values.
        """
        g1, g2, g3, X, Y = self._expand_dims(g1, g2, g3, X, Y)  # Expand dimensions for calculations
        return 1/2 * g3 * (1 - np.sqrt(g1 + g2*X + g3*Y))  # Calculate qy based on g1 and g2
        # return 1/2 * g3 * np.ones_like(Y)

    def k_func(self, g1, g2, g3, X, Y):
        """
        Computes a function 'k' based on inputs g1, g2 and g3.

        Args:
            g1 (array): First generator output.
            g2 (array): Second generator output.
            X (array): Meshgrid X coordinates.
            Y (array): Meshgrid Y coordinates.

        Returns:
            array: The computed k values.
        """
        g1, g2, g3, X, Y = self._expand_dims(g1, g2, g3, X, Y)  # Expand dimensions for calculations
        return np.sqrt(g1 + g2*X + g3*Y) * (1 - np.sqrt(g1 + g2*X + g3*Y))  # Calculate k (returns an array of ones)
        # return np.sqrt(g1 + g2*X + g3*Y)

    def f_func(self, g1, g2, g3, X, Y):
        """
        Computes a function 'k' based on inputs g1, g2 and g3.

        Args:
            g1 (array): First generator output.
            g2 (array): Second generator output.
            X (array): Meshgrid X coordinates.
            Y (array): Meshgrid Y coordinates.

        Returns:
            array: The computed k values.
        """
        g1, g2, g3, X, Y = self._expand_dims(g1, g2, g3, X, Y)  # Expand dimensions for calculations
        return (1/np.sqrt(g1 + g2*X + g3*Y)) * ((g2**2)/4 + (g3**2)/4)  # Calculate k (returns an array of ones)
        # return np.zeros_like(g1*X)

    def generate_dataset(self, test_size=0.2):
        """
        Generates a synthetic dataset based on the defined functions and splits it into training and validation sets.

        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split (default is 0.2).

        Returns:
            dict: A dictionary containing the training and validation datasets and their respective parameters.
        """
        # Calculate function outputs for u, qx, qy, and k
        u = self.u_func(self.g1, self.g2, self.g3, self.X_mesh, self.Y_mesh)
        qx = self.qx_func(self.g1, self.g2, self.g3, self.X_mesh, self.Y_mesh)
        qy = self.qy_func(self.g1, self.g2, self.g3, self.X_mesh, self.Y_mesh)
        k = self.k_func(self.g1, self.g2, self.g3, self.X_mesh, self.Y_mesh)
        f = self.f_func(self.g1, self.g2, self.g3, self.X_mesh, self.Y_mesh)

        # Concatenate relevant data for X input features
        X = np.concatenate(
            (u[:, :, 0, np.newaxis], u[:, :, self.N_DISCRETIZATION - 1, np.newaxis], 
             u[:, 0, :, np.newaxis], u[:, self.N_DISCRETIZATION - 1, :, np.newaxis],
             qx[:, :, 0, np.newaxis], qx[:, :, self.N_DISCRETIZATION - 1, np.newaxis], 
             qy[:, 0, :, np.newaxis], qy[:, self.N_DISCRETIZATION - 1, :, np.newaxis]),
            axis=2)  # Combine features into a single array along a new dimension
        y = u  # Output labels for training

        # Split the data into training and validation sets
        split_data = train_test_split(X, y, self.g1, self.g2, self.g3, qx, qy, u, k, f, test_size=test_size, random_state=self.custom_seed)
        (X_train, X_val, y_train, y_val, g1_train, g1_val, g2_train, g2_val, g3_train, g3_val, qx_train, qx_val, qy_train, qy_val, u_train, u_val, k_train, k_val, f_train, f_val) = split_data

        # Create a dictionary to hold the datasets and parameters
        data_dict = {
            "N_DATA": self.N_DATA,
            "N_DISCRETIZATION": self.N_DISCRETIZATION,
            "x0": self.x0,
            "xN": self.xN,
            "y0": self.y0,
            "yN": self.yN,
            "x": self.x,
            "x_step_size": self.x_step_size,
            "y": self.y,
            "y_step_size": self.y_step_size,
            "X_mesh": self.X_mesh,
            "Y_mesh": self.Y_mesh,
            "X_train": X_train, "X_val": X_val,
            "y_train": y_train, "y_val": y_val,
            "g1_train": g1_train, "g1_val": g1_val,
            "g2_train": g2_train, "g2_val": g2_val,
            "g3_train": g3_train, "g3_val": g3_val,
            "qx_train": qx_train, "qx_val": qx_val,
            "qy_train": qy_train, "qy_val": qy_val,
            "u_train": u_train, "u_val": u_val,
            "k_train": k_train, "k_val": k_val,
            "f_train": f_train, "f_val": f_val
        }

        return data_dict  # Return the dataset dictionary
    
    def save_data(self, data_dict):
        """
        Saves the generated dataset to a file in pickle format.

        Args:
            data_dict (dict): The dictionary containing the dataset to be saved.
        """
        create_folder(self.PATH)  # Ensure the directory exists for saving data

        base_pickle_path = os.path.join(self.PATH, f"{self.MODEL_NAME}.pkl")  # Base path for the pickle file
        pickle_path = base_pickle_path  # Initialize the path
        file_index = 1  # Initialize file index for versioning

        # Check if the file already exists and increment the file index if necessary
        while os.path.exists(pickle_path):
            pickle_path = os.path.join(self.PATH, f"{self.MODEL_NAME}_{file_index}.pkl")  # Update the path with the index
            file_index += 1  # Increment the index

        # Save the data dictionary to a pickle file
        with open(pickle_path, "wb") as f:
            pickle.dump(data_dict, f)  # Serialize and save the dictionary

        print(f"Data saved in {pickle_path}")  # Output confirmation of the save operation
