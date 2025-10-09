import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class DataAquisition:

    def __init__(self, data=None):
        self.basics_path_ = ("../title.basics.tsv")
        self.ratings_path_ = ("../data/title.ratings.tsv")
        #Check paths
        self.check_paths()

        if data is None:

            self.data = self.total_data()
            self.histodata = self.data[['Release year', 'Rating', 'Number of votes', 'Runtime']].to_numpy()
        else:
            self.data = data
            self.histodata = data

    
    """
     Exercise 1 ------------------------------- Exercise 1
    """
    
    @staticmethod
    def generate_2d_dist(size):
        # Triangular distribution 
        random_dist_triangular = np.random.triangular(left=0, mode=5, right=10, size=size)
        
        # Normal distribution 
        random_dist_normal = np.random.normal(loc=5, scale=2, size=size)
        
        # Combine the two distributions column-wise
        random_combined = np.column_stack((random_dist_triangular, random_dist_normal))
        
        return random_combined

    
    #Works for exercise 3 also
    def plot_histograms(self, bins=50):
        for i in range(self.histodata.shape[1]):
            plt.figure(figsize=(4, 6))
            plt.hist(self.histodata[:, i], bins=bins, density=True, alpha=0.7, color='blue')
            plt.title(f'Histogram of column {i}')         
            plt.xlabel(f'Values in column {i}')
            plt.ylabel('Density')
            plt.grid(True)
            plt.show()



    """
     Exercise 2 ------------------------------- Exercise 2
    """

    def plot_heatmap(self, bins=50, title='Heatmap of 2D Random Distribution'):
        plt.figure(figsize=(8, 6))
        plt.hist2d(self.histodata[:, 0], self.histodata[:, 1], bins=bins, cmap='hot')
        plt.colorbar(label='Frequency')
        plt.title(title)
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.grid(True)
        plt.show()



    """
     Exercise 3 ------------------------------- Exercise 3
    """

    def check_paths(self):
        for path in [self.basics_path_, self.ratings_path_]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            else:
                print(f"File found: {path}")

    
    
    #Function to read selected columns from a TSV file. Also include the option to limit the number of rows read, used for finding the rows I need.
    @staticmethod
    def read_file_tsv(filepath, columns=None, nrows=5):

        return pd.read_csv(filepath, sep='\t', usecols=columns, nrows=nrows)
    
    def clean_data(self, compiled_df):
        # Remove rows with missing or non-numeric 'startYear'
        compiled_df['startYear'] = pd.to_numeric(compiled_df['startYear'], errors='coerce') # Convert to numeric, setting errors to NaN
        compiled_df['averageRating'] = pd.to_numeric(compiled_df['averageRating'], errors='coerce') # Convert to numeric, setting errors to NaN
        compiled_df['numVotes'] = pd.to_numeric(compiled_df['numVotes'], errors='coerce') # Convert to numeric, setting errors to NaN
        compiled_df['runtimeMinutes'] = pd.to_numeric(compiled_df['runtimeMinutes'], errors='coerce') # Convert to numeric, setting errors to NaN

        compiled_df = compiled_df[compiled_df['runtimeMinutes'] < 2000].copy() # Remove unrealistic and outlying runtimes
        # Calculate number of words in the title
        compiled_df['Words in title'] = compiled_df['primaryTitle'].astype(str).apply(lambda x: len(x.split()))

        # Calculate number of letters in the title, without spaces and special characters
        compiled_df['Length of title'] = compiled_df['primaryTitle'].astype(str).apply(lambda x: sum(c.isalpha() for c in x))

        compiled_df = compiled_df.rename(columns={
            'tconst': 'Title ID',
            'startYear': 'Release year',
            'averageRating': 'Rating',
            'numVotes': 'Number of votes',
            'primaryTitle': 'Title',
            'titleType': 'Type',
            'runtimeMinutes': 'Runtime',
            'genres': 'Genres'
        })
        return compiled_df.dropna(subset=['Release year', 'Rating', 'Number of votes', 'Runtime']).reset_index(drop=True)

    def total_data(self):
        data_b = self.read_file_tsv(self.basics_path_, columns=None, nrows=None)
        data_r = self.read_file_tsv(self.ratings_path_,columns=None, nrows=None)
        merged_data = pd.merge(data_b, data_r, on='tconst')
        merged_data = merged_data[merged_data['numVotes'] >= 200]
        merged_data = self.clean_data(merged_data)


        return merged_data

    """
     Exercise 4 ------------------------------- Exercise 4
    """
    @staticmethod
    def calc_pmf(data):
        values, counts = np.unique(data, return_counts=True)
        pmf = counts / counts.sum()
        return values, pmf
    
    @staticmethod
    def plot_discrete(data, column=0):
        values, pmf = DataAquisition.calc_pmf(data[:, column])
        plt.figure(figsize=(6, 4))
        plt.stem(values, pmf)
        plt.title(f'PMF of Column {column}')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.show()

    def plot_all_pmfs(self):
        for i in range(self.histodata.shape[1]):
            try:

                DataAquisition.plot_discrete(self.histodata, column=i)
            except: 
                print(f"Could not plot PMF for column {i}. Data might be non-numeric.")



    """
     Exercise 5 ------------------------------- Exercise 5
    """
    
    @staticmethod
    def calc_cdf(data):
        values, pmf = DataAquisition.calc_pmf(data)
        cdf = np.cumsum(pmf)
        return values, cdf
    
    def plot_all_cdfs(self):
        for i in range(self.histodata.shape[1]):
            try:
                values, cdf = DataAquisition.calc_cdf(self.histodata[:, i])
                plt.stem(values, cdf)
                plt.title(f'CDF of Column {i}')
                plt.xlabel('Value')
                plt.ylabel('Probability')
                plt.grid(True)
                plt.show()

            except:
                print(f"Could not compute CDF for column {i}. Data might be non-numeric.")

    def cdf(self, column=0):

        values, cdf = DataAquisition.calc_cdf(self.histodata[:, column])
        return values, cdf