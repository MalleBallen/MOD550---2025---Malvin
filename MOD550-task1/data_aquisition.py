import matplotlib.pyplot as plt
import numpy as np


class DataAquisition:

    def __init__(self, data):
        self.data = data

    #Works for exercise 3 also
    def plot_histograms(self, bins=50):
        for i in range(self.data.shape[1]):
            plt.figure(figsize=(4, 6))
            plt.hist(self.data[:, i], bins=bins, density=True, alpha=0.7, color='blue')
            plt.title(f'Histogram of Column {i}')
            plt.xlabel(f'Values in Column {i}')
            plt.ylabel('Density')
            plt.grid(True)
            plt.show()


    def plot_heatmap(self, bins=50, title='Heatmap of 2D Random Distribution'):
        plt.figure(figsize=(8, 6))
        plt.hist2d(self.data[:, 0], self.data[:, 1], bins=bins, cmap='hot')
        plt.colorbar(label='Frequency')
        plt.title(title)
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.grid(True)
        plt.show()


    def plot_pmf(self, column=0):
        values, counts = np.unique(self.data[:, column], return_counts=True)
        pmf = counts / counts.sum()
        plt.figure(figsize=(6, 4))
        plt.stem(values, pmf, use_line_collection=True)
        plt.title(f'PMF of Column {column}')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.show()


