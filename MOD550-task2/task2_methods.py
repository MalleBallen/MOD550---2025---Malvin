import sys
sys.path.append('../MOD550-task1') # Adjust the path to find data_aquisition module
from data_aquisition import DataAquisition as aq


import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
#import tensorflow  
#from tensorflow import keras
#from tensorflow.keras import layers


class DataModel:
    """
    DataModel is a class for loading, preprocessing, analyzing, and modeling movie data.
    It gives methods for splitting data, feature extraction, visualization, linear and neural network regression,
    clustering with KMeans, and evaluating model performance using metrics such as Mean Squared Error (MSE).
    """

    def __init__(self, rows_from_data=None):

        """
        Initializes the DataModel by loading data from the DataAquisition class.
        Splits the data into training, validation, and test sets, and extracts features and targets for each set.
        """
        if rows_from_data is not None:
            self.data = aq(rows_from_data=rows_from_data).data
        else:
            self.data = aq().data
        # self.data_train, self.data_val, self.data_test = self.train_val_test_split(train_ratio=0.7, val_ratio=0.15, seed=1) # 70% train, 15% val, 15% test
        #  self.X_train, self.y_train = self.extract_xy(self.data_train)
        #  self.X_val, self.y_val = self.extract_xy(self.data_val)
        #  self.X_test, self.y_test = self.extract_xy(self.data_test)

    

    def train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, seed=1):
        """
        Splits the dataset into train, validation, and test sets after shuffling.
        Args:
            train_ratio: Proportion of data to use for training.
            val_ratio: Proportion of data to use for validation.
            seed (int, optional): Random seed for reproducibility.
        Returns:
            tuple: (train_data, val_data, test_data) as lists of row dicts (vanilla python). NOT dataframes.
        """
        if train_ratio + val_ratio > 1:
            raise ValueError("Ratios are over 1")
        if seed is not None:
            random.seed(seed)
            
        # Convert column dict to list of row dicts
        columns = list(self.data.keys())
        n_rows = len(self.data[columns[0]])
        rows = [ {col: self.data[col][i] for col in columns} for i in range(n_rows) ]
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_rows = rows[:n_train]
        val_rows = rows[n_train:n_train + n_val]
        test_rows = rows[n_train + n_val:]
        return train_rows, val_rows, test_rows
    
    def extract_xy(self, data):
        """
        Extracts features (X) and target (y) arrays from a list of data dictionaries.

        Args:
            data (list of dict): Each dict must contain the keys 'Runtime', 'Release year', 'Words in title', 'Length of title', and 'Rating'.
                All values should be convertible to float or int. 
                Used to create feature matrix X and target vector y.

        Returns:
            tuple: (X, y) of numpy arrays
                X: 2D numpy array of shape (n_samples, 4) with features in the order [Runtime, Release year, Words in title, Length of title].
                y : numpy array with target ratings.
        """

        x = np.array([
            [row['Runtime'], row['Release year'], row['Words in title'], row['Length of title']]
            for row in data
        ])
        y = np.array([row['Rating'] for row in data])
        return x, y
    


    def scatter_plots_features_vs_rating(self):
        """
        Creates scatter plots using hexbin for each feature (Runtime, Release year, Words in title, Length of title)
        against Rating.
        """
        
        # Convert column dict to list of row dicts
        columns = list(self.data.keys())
        n_rows = len(self.data[columns[0]])
        data_rows = [{col: self.data[col][i] for col in columns} for i in range(n_rows)]

        features = ['Runtime', 'Release year', 'Words in title', 'Length of title']
        ratings = [row['Rating'] for row in data_rows]

        plt.figure(figsize=(16, 4))
        for i, feat in enumerate(features):
            plt.subplot(1, 4, i + 1)
            x = [row[feat] for row in data_rows]
            y = ratings
            plt.hexbin(x, y, gridsize=40, cmap='viridis', mincnt=1) # Hexbin for better visualization with many points
            plt.xlabel(feat)
            plt.ylabel('Rating')
            plt.title(f'{feat} vs Rating')
            plt.colorbar(label='Counts')
        plt.tight_layout()
        plt.show()

    def linear_regression(self, x, y):
        """
        Performs simple single variate linear regression using the least squares method.
        Args:
            x : The independent variable values.
            y : The dependent variable values.
        Returns:
            tuple: Intercept (b0) and slope (b1) of the regression line.
        """
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        b1 = numerator / denominator
        b0 = y_mean - b1 * x_mean
        return b0, b1

    def predict(self, x_new, b0, b1):
        """
        Predicts the dependent variable value for a new variable input using the linear regression coefficients.
        Args:
            x_new: The new value of the variable.
            b0: Intercept of the regression line.
            b1: Slope of the regression line.
        Returns:
            Predicted value of the variable.
        """
        return b0 + b1 * x_new
    
    def predict_linear_regression(self, x_new, x=None, y=None):
        """
        Predicts the dependent variable value for a new independent variable input using linear regression.
        If x and y are not provided, uses the data loaded in self.data, namely 'Release year' as x and 'Rating' as y.
        Args:
            x_new: The new value of the independent variable to predict for.
            x: variable values.
            y: target variable values.
        Returns:
            Predicted value of the target variable for x_new.
        """
        if x is None and y is None:
            x = self.data['Release year']
            y = self.data['Rating']
        elif x is None or y is None:
            raise ValueError("Both x and y must be provided if one is provided.")
        b0, b1 = self.linear_regression(x, y)
        return self.predict(x_new, b0, b1)


    def mean_squared_error(self, actual, predicted):
        """
        Computes the Mean Squared Error (MSE) between actual and predicted values.
        Args:
            actual: True values.
            predicted: Predicted values.
        Returns:
            The mean squared error.
        """
        n = len(actual)
        squared_errors = 0
        for i in range(n):
            squared_errors += (actual[i] - predicted[i]) ** 2
        mse = squared_errors / n
        return mse
    

    def test_MSE(self):
        """
        Fits a linear regression model on the training data and evaluates its performance on the test data using Mean Squared Error (MSE).

        Returns:
            tuple: (b0, b1, mse)
                b0: Intercept of the regression line fitted on the training data.
                b1: Slope of the regression line fitted on the training data.
                mse: Mean Squared Error of the predictions on the test data.
        """

        data_train, unused , data_test = self.train_val_test_split(train_ratio=0.7, val_ratio=0, seed=1) #To get 30 percent test data)

        # Extract x (Release year) and y (Rating) from train data
        x_train = [row['Release year'] for row in data_train]
        y_train = [row['Rating'] for row in data_train]

        # Fit linear regression on train data
        b0, b1 = self.linear_regression(x_train, y_train)
        x_test = [row['Release year'] for row in data_test]
        y_test = [row['Rating'] for row in data_test]

        # Predict on test data
        y_pred = [self.predict(x, b0, b1) for x in x_test]

        # Calculate MSE
        mse = self.mean_squared_error(y_test, y_pred)
        return b0, b1, mse


    
    def make_nn(self, input_dim, layers_units=[64, 32], activation='relu', output_activation='linear', optimizer='adam', loss='mse'):
        """
        Creates and compiles a neural network using Keras.
        
        Args:
            input_dim: Number of input features.
            layers_units: List with the number of units in each hidden layer.
            activation: Activation function for hidden layers. Using same for all hidden layers.
            output_activation: Activation function for output layer.
            optimizer: Optimizer to use.
            loss: Loss function to use.
            
        Returns:
            keras.Model: Compiled Keras model.
        """
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers
        for units in layers_units:
            model.add(layers.Dense(units, activation=activation)) 

        model.add(layers.Dense(1, activation=output_activation))
        model.compile(optimizer=optimizer, loss=loss)
        return model
    
    def evaluate_nn_architectures(self, X_train, y_train, X_val, y_val, architectures, epochs=25, batch_size=128):
        """
        Evaluates multiple neural network architectures on the provided training and validation data.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target data.
            X_val (np.ndarray): Validation feature data.
            y_val (np.ndarray): Validation target data.
            architectures: List of architectures, where each architecture is a list specifying the number of units in each hidden layer.
            epochs: Number of epochs to train each model. 
            batch_size: Batch size for training.

        Returns:
            list: Sorted list of dictionaries, each containing:
            - 'architecture': The architecture tested.
            - 'val_mse': The final validation mean squared error for that architecture.
        """
        results = []
        for arch in architectures:
            model = self.make_nn(input_dim=X_train.shape[1], layers_units=arch)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
            val_mse = history.history['val_loss'][-1] #Using the last epoch's validation loss as the metric
            results.append({'architecture': arch, 'val_mse': val_mse})

            results.sort(key=lambda x: x['val_mse'], reverse=True)

        print("All architectures:")
        for res in sorted(results, key=lambda x: x['val_mse']):
            print(f"  Architecture: {res['architecture']}, Validation MSE: {res['val_mse']}")
 

    def test_nn(self, architectures=[[64, 32]], epochs=20, batch_size=64):
        """
        Fits and evaluates neural network architectures on the training and validation data from init.

        Args:
            architectures: List of architectures, where each architecture is a list specifying the number of units in each hidden layer. 
            epochs: Number of epochs to train each model. 
            batch_size: Batch size for training. 

            Prints the validation MSE for each architecture.
        """
        self.evaluate_nn_architectures(self.X_train, self.y_train, self.X_val, self.y_val, architectures=architectures, epochs=epochs, batch_size=batch_size)


    def fit_kmeans(self, data, n_clusters=3, n_init=40):
        """
        Fits a KMeans clustering model to the provided data using scikit-learn.

        Args:
            data: Feature data to cluster.
            n_clusters: Number of clusters to form.
            n_init: Number of time the k-means algorithm will be run with different centroid seeds.

        Returns:
            KMeans: Fitted KMeans model.
        """
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=n_init, random_state=0)
        kmeans.fit(data)
        return kmeans

    def kmeans_cluster_centers(self, n_clusters=3, n_init=40):
        """
        Fits a KMeans clustering model to the training feature data and displays the cluster centers as a DataFrame.

        Args:
            n_clusters: Number of clusters to form. 
            n_init: Number of initializations to perform.
        
        Displays a DataFrame of cluster centers with feature names as columns.
        """

        kmeans = self.fit_kmeans(self.X_train, n_clusters=n_clusters, n_init=n_init)
        headers = ['Runtime', 'Release year', 'Words in title', 'Length of title']
        centers = kmeans.cluster_centers_
        df_centers = pd.DataFrame(centers, columns=headers) # Create DataFrame with headers for better display
        print(df_centers)

    def linear_regression_multivariate(self, X=None, y=None, sklearn=False):
        """
        Performs multivariate linear regression using the least squares method.
        Args:
            X: The variable values.
            y: The target variable values.
        Returns:
            numpy array: Coefficients of the regression line (including intercept if added to X).
        """

        if X is None and y is None:
            X = self.data[['Runtime', 'Release year', 'Words in title', 'Length of title']].to_numpy()
            y = self.data['Rating'].to_numpy()
        elif X is None or y is None:
            raise ValueError("Both X and y must be provided if one is provided.")
        
        
        if not sklearn:
            # Calculate beta using the normal equation, given in presentation 008. Gives intercept at 0.
            beta = np.linalg.inv(X.T @ X) @ X.T @ y


        else: 
            # Using sklearn library for linear regression, which allows intercept another place than 0. 
            linreg = LinearRegression(fit_intercept=True)
            linreg.fit(X, y)
            beta = np.append(linreg.intercept_, linreg.coef_)  # Combine intercept and coefficients

        return beta
    
    def predict_y_from_x_beta(self, X, beta, intercept=True):
        """
        Predicts the dependent variable values given the feature matrix X and regression coefficients beta.
        Args:
            X (numpy array): Feature values.
            beta (numpy array): Regression coefficients. If intercept=True, beta[0] is the intercept.
            intercept (bool): Whether the first element of beta is the intercept.
        Returns:
            numpy.ndarray: Predicted values.
        """
        if intercept:
            y = X @ beta[1:] +beta[0]

        else:
            y = X @ beta
        return y


    def test_linreg_multivariate(self, sklearn=False):
        """
        Fits a multivariate linear regression model on the training data and evaluates its performance on the test data using Mean Squared Error (MSE).

        Args:
            sklearn (bool): If True, uses scikit LinearRegression (with intercept). If false, uses vanilla numpy (no intercept).

        Returns:
            Prints the model coefficients and the MSE on the test data.
        """
        
        beta = self.linear_regression_multivariate(self.X_train, self.y_train, sklearn=sklearn)

        y_predicted = self.predict_y_from_x_beta(self.X_test, beta, intercept=sklearn) #If sklearn=True, then intercept is true

        mse_from_model = self.mean_squared_error(self.y_test, y_predicted)

        if sklearn:
            print(f"Intercept: {beta[0]}, Coefficients: {beta[1:]}")
        else:
            print(f"Intercept: 0, Coefficients: {beta}")
        print(f"This gives an MSE of: {mse_from_model}")