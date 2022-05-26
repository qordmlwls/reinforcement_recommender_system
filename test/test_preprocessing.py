import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.module_for_data_preparation import DataPreparation

import pandas as pd

if __name__ == '__main__':

    try:
        movie_df = pd.read_csv(os.path.join('..','mydataset','movies.csv'))
        path = '..'
    except FileNotFoundError:
        movie_df = pd.read_csv(os.path.join(os.getcwd(), 'mydataset', 'movies.csv'))
        path = os.getcwd()
    rating_df = pd.read_csv(os.path.join(path, 'mydataset', 'ratings.csv'))

    ##preprocessing
    preparationer = DataPreparation(path)
    preparationer.run(movie_df, rating_df)