import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.module_for_real_time_prediction import ContentRank
import pandas as pd

if __name__ == '__main__':
    try:
        prediction_data_example = pd.read_csv(os.path.join('..', 'mydataset', 'predoction_data_example.csv'))
        path = '..'
    except FileNotFoundError:
        prediction_data_example = pd.read_csv(os.path.join(os.getcwd(), 'mydataset', 'predoction_data_example.csv'))
        path = os.getcwd()

    rank_generate = ContentRank(path)
    rank_df = rank_generate.run(prediction_data_example)
    rank_df.to_csv(os.path.join(path, 'output', 'rank_df.csv'), mode='w', encoding='utf-8-sig')

