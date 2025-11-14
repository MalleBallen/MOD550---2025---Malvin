from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

import numpy as np

from scipy.stats import beta

from matplotlib import pyplot as plt   
import seaborn as sns
import sys
sys.path.append('../MOD550-task2')
from task2_methods import DataModel 

class Methods:
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
        features = ['Runtime', 'Release year', 'Words in title', 'Length of title']
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(self.data[features], self.data['Rating'], test_size=0.3, random_state=1)
        scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        return X_train_norm, y_train_reg, X_test_norm, y_test_reg
    
    def print_success_rate(self):
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

    def create_clfs(self, kernels):
        trained_clfs = {}
        for kernel_name, clf in kernels:
            print(f"Training - Kernel: {kernel_name}")
            clf.fit(self.X_train, self.y_train_bin)
            trained_clfs[kernel_name] = clf

        return trained_clfs

    def create_grid_search(self):
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

        labels = ['Low Rating', 'High Rating']

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.show()

    def eval_parameters(self, y_pred, y_true):

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        return accuracy, precision, recall, conf_matrix
    
    def print_evaluation(self, accuracy, precision, recall, conf_matrix, name):
        print(f"Evaluation Metrics for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        self.plot_conf_mat(conf_matrix, title=f'Confusion Matrix - {name}')


    def evaluate_models(self, trained_models):
        for kernel_name, clf in trained_models.items():
            y_pred = clf.predict(self.X_test)
            eval_params = self.eval_parameters(y_pred, self.y_test_bin)
            self.print_evaluation(*eval_params, name=f"Kernel: {kernel_name}")

    def linear_regression_multivariate(self):

        # Using sklearn library for linear regression, which allows intercept another place than 0. 
        linreg = LinearRegression(fit_intercept=True)
        linreg.fit(self.X_train, self.y_train_reg)
        beta = np.append(linreg.intercept_, linreg.coef_)  # Combine intercept and coefficients

        return beta
    
    def predict_multivariate(self):
        beta = self.linear_regression_multivariate()
        return  self.X_test @ beta[1:] + beta[0]
    
    def nn_classifier(self, hidden_layers, activation ,solver):
        nn = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, max_iter=300, validation_fraction=0.2)
        nn.fit(self.X_train, self.y_train_bin)
        return nn
    
 
