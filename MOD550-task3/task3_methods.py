from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.preprocessing import StandardScaler                    # For feature normalization
from sklearn import svm                                             # For Support Vector Machine models
from sklearn.svm import SVC                                         # For SVM classifier specifically
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix  # For model evaluation metrics
from sklearn.linear_model import LinearRegression                   # For linear regression modeling
from sklearn.neural_network import MLPClassifier                    # For neural network classification

import numpy as np                                                  # For numerical operations and array handling

from scipy.stats import beta                                        # For Beta distribution (Bayesian analysis)

from matplotlib import pyplot as plt                                # For plotting graphs and figures

import seaborn as sns                                               # For advanced data visualization (heatmaps)
import sys                                                          # For system-specific parameters and functions
import pymc as pm                                                   # For Bayesian statistical modeling

sys.path.append('../MOD550-task2')                                  # To import modules from another directory
from task2_methods import DataModel                                 # Custom data loading and preprocessing class

class Methods:
    """
    The Methods class provides a collection of methods for data preprocessing, 
    statistical modeling, machine learning, and evaluation. It includes functionalities 
    for normalizing data, performing linear regression (both frequentist and Bayesian), 
    training and evaluating Support Vector Machines (SVMs) with different kernels, 
    and implementing neural network classifiers.

    Key Features:
    - Data normalization and preprocessing.
    - Linear regression (frequentist and Bayesian approaches).
    - Beta-Binomial posterior computation and visualization.
    - Support Vector Machine (SVM) training with hyperparameter tuning.
    - Neural network classifier training and evaluation.
    - Model evaluation metrics including accuracy, precision, recall, and confusion matrix visualization.

    Attributes:
    - DM: Instance of DataModel for data loading and preprocessing.
    - data: Preprocessed dataset.
    - kernels: List of kernel types for SVM.
    - X_train, y_train_reg, X_test, y_test_reg: Training and testing datasets (normalized).
    - y_train_bin, y_test_bin: Binary labels for classification tasks.
    - param_grid, param_grid_linear: Hyperparameter grids for SVM tuning.
    """

    def __init__(self):
        self.DM = DataModel(rows_from_data=1500000) # Load only 1.5 million rows for faster computation. 
        #Not all IDs have ratings, therefore some are deleted.
        self.data = self.DM.data.copy()
        self.kernels =[
        ('linear'),
        ('rbf')
        ]

        self.X_train, self.y_train_reg, self.X_test, self.y_test_reg = self.normalize()
        self.y_train_bin = (self.y_train_reg >= 7).astype(int)
        self.y_test_bin = (self.y_test_reg >= 7).astype(int)
            
 
        self.param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }

        self.param_grid_linear = { # Because linear kernel does not use gamma
        'C': [0.1, 1, 10, 100]
    }

    def normalize(self):
        """
        Normalize the selected features of the dataset using standard scaling.
        This method splits the data into training and testing sets, then applies
        standard scaling to normalize the features by removing the mean and scaling
        to unit variance.
        Returns:
            A tuple containing:
                - X_train_norm : Normalized training feature set.
                - y_train_reg : Training target values.
                - X_test_norm: Normalized testing feature set.
                - y_test_reg: Testing target values.
        """

        features = ['Runtime', 'Release year', 'Words in title', 'Length of title']
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(self.data[features], self.data['Rating'], test_size=0.3, random_state=1)
        scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        return X_train_norm, y_train_reg, X_test_norm, y_test_reg
    
    def print_success_rate(self):
        """
        Calculate and print the success rate of the training data.
        This method computes the number of successes and trials from the binary 
        training labels (`y_train_bin`) and prints the success rate as a formatted 
        string.
        Attributes:
            self.successes (int): The total number of successes in the training data.
            self.trials (int): The total number of trials in the training data.
        Output:
            Prints the number of successes, trials, and the probability of success.
        """

        self.successes = sum(self.y_train_bin)
        self.trials = len(self.y_train_bin)
        print(f"Successes: {self.successes}, Trials: {self.trials} and P(success)={self.successes/self.trials:.3f}")
    
    def compute_posteriors(self, priors):
        """
        Compute Beta-Binomial posteriors for multiple prior parameter sets.
        
        Parameters:
        priors (list of tuples): Each tuple is (alpha_prior, beta_prior)
        successes (int): Observed successes
        trials (int): Total trials
        
        Returns:
        list of dicts: Each dict contains posterior info for one prior
        """
        results = []
        for alpha_prior, beta_prior in priors:
            alpha_post = alpha_prior + self.successes
            beta_post = beta_prior + self.trials - self.successes
            posterior_mean = alpha_post / (alpha_post + beta_post)
            results.append({
                "priors" : [alpha_prior, beta_prior],
                "posteriors": [alpha_post, beta_post],
                "posterior_mean": posterior_mean,
            })
        return results
    
    def plot_posteriors(self, results):
        """
        Plot Beta distributions for each posterior in a list of dictionaries.
        
        Parameters:
        results (list of dict): Each dict must have keys:
            - "priors": [alpha_prior, beta_prior]
            - "posteriors": [alpha_post, beta_post]
            - "posterior_mean": float

        """
        p = np.linspace(0, 1, 500)
        plt.figure(figsize=(12, 7))

        for i, res in enumerate(results):
            alpha_post, beta_post = res["posteriors"]
            mean = res["posterior_mean"]
            alpha_prior, beta_prior = res["priors"]
            pdf = beta.pdf(p, alpha_post, beta_post)
            label = f'Beta({alpha_post},{beta_post}). Mean={mean:.3f}. Prior: Beta({alpha_prior},{beta_prior})'
            plt.plot(p, pdf, linewidth=2, label=label)

        plt.title('Posterior Beta Distributions')
        plt.xlabel('Probability p')
        plt.ylabel('Density')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.show()
    
    def linear_regression_multivariate(self):
        
        # Using sklearn library for linear regression, which allows intercept another place than 0. 
        linreg = LinearRegression(fit_intercept=True)
        linreg.fit(self.X_train, self.y_train_reg)
        beta = np.append(linreg.intercept_, linreg.coef_)  # Combine intercept and coefficients

        return beta
    
    def predict_multivariate_b(self, beta): 
        return  self.X_test @ beta[1:] + beta[0]
    

    def predict_multivariate(self):
        beta = self.linear_regression_multivariate()
        return self.predict_multivariate_b(beta)
    

    def compute_mse(self, beta):
        """
        Compute Mean Squared Error (MSE) for given beta coefficients.

        Parameters:
        beta: Coefficient vector (beta[0] intercept and beta-params for features)
        y: Target vector 
        """

        y = np.array(self.y_test_reg)

        y_pred = self.predict_multivariate_b(beta)  

        mse = np.mean((y - y_pred) ** 2)
        print(f"Mean Squared Error (MSE): {mse:.3f} for beta coefficients: {beta}")

    def bayesian_regression_multivariate(self):
        """
        Perform Bayesian regression for a multivariate dataset using pymc.
        This method constructs a Bayesian regression model with a normal prior 
        for the intercept and coefficients, and a half-normal prior for the 
        standard deviation of the residuals. The model is trained using the 
        Sequential Monte Carlo sampler.
        Inspiration is gathered from lecture notes and lecture jupyter file.


        Attributes:
            self.X_train (numpy.ndarray): The training data features.
            self.y_train_reg (numpy.ndarray): The training data target values 
                for regression.
                
        The method prints the summary of the posterior distribution 
        of the model parameters.

        """

        with pm.Model() as model:

            b0 = pm.Normal("b0", mu=0, sigma=100)  # Intercept
            b = pm.Normal("b", mu=0, sigma=[10, 10, 10, 10], shape=4)  
            sigma = pm.HalfNormal("sigma", sigma=10)

            # Linear predictor
            mu = b0 + pm.math.dot(self.X_train, b)

            # Likelihood
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y_train_reg)

            # Sequential Monte Carlo sampler
            trace = pm.sample_smc(draws=4000, progressbar=True)

        print(pm.summary(trace))

    def create_clfs(self, kernels):
        """
        Trains and returns a dictionary of classifiers using the provided kernels.
        Args:
            kernels (list of tuples): A list of tuples where each tuple contains:
                - kernel_name: The name of the kernel.
                - clf: The classifier object to be trained.
        Returns:
            dict: A dictionary where the keys are kernel names and the values 
            are the trained classifier objects.
        """

        trained_clfs = {}
        for kernel_name, clf in kernels:
            print(f"Training - Kernel: {kernel_name}")
            clf.fit(self.X_train, self.y_train_bin)
            trained_clfs[kernel_name] = clf

        return trained_clfs

    def create_grid_search(self):
        """
        Perform grid search optimization for different SVM kernels and return the best models.
        This method iterates over the specified kernel types, performs a grid search 
        to find the best hyperparameters for each kernel, and creates an optimized 
        SVM model using the best parameters, based on accuracy. 
        Returns:
         list of tuples where each tuple contains:
                - kernel_name: The name of the kernel.
                - best_model: The optimized SVM model for the kernel.
        Notes:
            - The method uses `GridSearchCV` to perform the grid search.
            - The training data is limited to the first 1400 samples of `self.X_train` 
              and `self.y_train_bin` for computational efficiency.
        """

        optimized_kernels = []
        
        for kernel_name in self.kernels:
            
            if kernel_name == 'linear':
                grid = GridSearchCV(SVC(kernel='linear'), self.param_grid_linear, cv=3, n_jobs=-1,verbose=2)
            else:
                grid = GridSearchCV(SVC(kernel=kernel_name), self.param_grid, cv=3, n_jobs=-1,verbose=2)

            grid.fit(self.X_train[:1400], self.y_train_bin[:1400])
            
            print(f"Best parameters for kernel {kernel_name}: {grid.best_params_}")
            print(f"Best score for kernel {kernel_name}: {grid.best_score_}")
            
            best_gamma = grid.best_params_.get('gamma', 0.0)
            # Create a new SVC clf with best params
            best_model = svm.SVC(kernel = kernel_name, C=grid.best_params_['C'], gamma=best_gamma, random_state=1)
            optimized_kernels.append((kernel_name, best_model))
        
        return optimized_kernels
    
    def plot_conf_mat(self, conf_matrix, title):
        """
        Plots a confusion matrix as a heatmap with annotations.
        Args:
            conf_matrix : The confusion matrix to be visualized.
            title: The title for the plot.
        
        Displays the confusion matrix plot.
        """

        labels = ['Low Rating', 'High Rating']

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.show()

    def eval_parameters(self, y_pred, y_true):
        """
        Creates an evaluation based on predicted and true y-values.

        args:
            y_pred: predicted y-vals
            y_true: actual y-vals

        returns:
            accuracy, precision, recall, conf_matrix
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        return accuracy, precision, recall, conf_matrix
    
    def print_evaluation(self, accuracy, precision, recall, conf_matrix, name):
        """
        Prints evaluation metrics and plots the confusion matrix for a given model.
        Args:
            accuracy, precision, recall, conf_matrix: Evaluation metrics and confusion matrix.

        """

        print(f"Evaluation Metrics for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        self.plot_conf_mat(conf_matrix, title=f'Confusion Matrix - {name}')


    def evaluate_models(self, trained_models):
        """
        Evaluates the performance of trained models on the test dataset.
        Args:
            trained_models (dict): A dictionary where keys are kernel names (or model identifiers) 
                                   and values are trained classifier objects.

        """
        
        for kernel_name, clf in trained_models.items():
            y_pred = clf.predict(self.X_test)
            eval_params = self.eval_parameters(y_pred, self.y_test_bin)
            self.print_evaluation(*eval_params, name=f"Kernel: {kernel_name}")


    def nn_classifier(self, hidden_layers, activation ,solver):
        """
        Trains a neural network classifier using the specified parameters.
        Args:
            hidden_layers: The number of neurons in each hidden layer.
            activation: Activation function for the hidden layers.
            solver: The solver for weight optimization.
        Returns:
            The trained neural network classifier.
        """
        
        nn = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, max_iter=300, validation_fraction=0.2)
        nn.fit(self.X_train, self.y_train_bin)
        return nn
    
    def nn_evalution(self, activation_functions, hidden_layers=(64, 32), solver='adam'):
        """
        Evaluates a neural network classifier with different activation functions.
        Args:
            hidden_layers: The number of neurons in each hidden layer.
            activation: Activation function for the hidden layers.
            solver: The solver for weight optimization.

        """

        for activation in activation_functions:
            nn_model = self.nn_classifier(hidden_layers=hidden_layers, activation=activation, solver=solver)
            y_pred_nn = nn_model.predict(self.X_test)
            evaluation_nn = self.eval_parameters(y_pred=y_pred_nn, y_true=self.y_test_bin)
            self.print_evaluation(*evaluation_nn, name=f"Neural Network Classifier ({activation} activation)")
        
    
