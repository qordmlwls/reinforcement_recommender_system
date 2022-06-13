import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import datetime
import pickle
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .module_for_generating_embedding import Model, get_embedding
import torch
import dateutil.parser


class DataPreparation:
    def __init__(self, path):
        self.recommend_limit = 5000
        self.pretrained_model = None
        self.args = {
            'random_seed': 42,  # Random Seed
            'pretrained_model': "beomi/KcELECTRA-base",  # Transformers PLM name
            'pretrained_tokenizer': "beomi/KcELECTRA-base",
            # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
            'batch_size': 32,
            'lr': 5e-6,  # Starting Learning Rate
            'epochs': 5,  # Max Epochs
            'max_length': 150,  # Max Length input size
            'train_data_path': "../input/SNS_content_classification_train.csv",  # Train Dataset file
            'val_data_path': '../input/test_prediction_pl.csv',  # Validation Dataset file
            'test_mode': False,  # Test Mode enables `fast_dev_run`
            'optimizer': 'AdamW',  # AdamW vs AdamP
            'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
            'fp16': True,  # Enable train on FP16
            'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
            'cpu_workers': os.cpu_count(),
        }
        self.path = path

    def refine_behavioral_data(self, behavioral_data, recommend_limit):
        """
        행동데이터 정제 후 로컬 저장, mapping_df(label_encoding data를 실제 content_id로 바꾸는 데 필요) 생성
        """
        mydf2 = behavioral_data[['userId', 'movieId', 'rating', 'timestamp']]
        num_recommend = [i for i in range(recommend_limit)]
        mydf2.columns = ['userId', 'movieId', 'rating', 'timestamp']
        le = LabelEncoder()
        le.fit(mydf2.movieId)
        mydf2['book_id'] = le.transform(mydf2['movieId'])
        mydf2['book_id'] = mydf2['book_id'].astype(int)
        mydf2 = mydf2.loc[mydf2['book_id'].isin(num_recommend), :]  ## 5000개 이하만
        tmp_df = mydf2[['book_id', 'movieId']]
        tmp_df.drop_duplicates(subset=['book_id'], inplace=True)
        mydf3 = mydf2[['userId', 'rating', 'timestamp', 'book_id']]
        mydf3.columns = ['reader_id', 'liked', 'when', 'book_id']
        # ### str일 경우
        # # mydf3['when'] = mydf3['when'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M"))
        # mydf3['when'] = mydf3['when'].apply(lambda x: dateutil.parser.parse(x))
        # ### datetime일경우 바로
        # mydf3['when'] = mydf3['when'].apply(lambda x: datetime.datetime.strftime(x, "%m/%d/%Y, %H:%M:%S"))
        ### timestamp
        mydf3['when'] = mydf3['when'].apply(
            lambda x: datetime.datetime.fromtimestamp(int(x)).strftime("%m/%d/%Y, %H:%M:%S"))
        ## 학습 데이터 로컬 저장
        mydf3.to_csv(os.path.join(self.path, 'mydataset', 'mydf.csv'), index=False)
        return tmp_df

    def generate_save_embeddings(self, tmp_dic, movie_df):
        """
        학습에 필요한 임베딩을 저장하는 함수
        """
        movie_df['document'] = movie_df['title'] + movie_df['genres']
        # tmp_dic2 = dict([(a, b) for a, b in zip(content_data['content_id'], content_data['embeddings'])])
        my_embeddings = dict()
        for key, value in tmp_dic.items():
            my_embeddings[key] = get_embedding(
                str(movie_df.loc[movie_df['movieId'] == value, ['document']]['document'].iloc[0]),
                self.pretrained_model)
        try:
            with open(os.path.join(os.getcwd(), 'mydataset', 'myembeddings.pickle'), 'wb') as handle:
                pickle.dump(my_embeddings, handle)
        except FileNotFoundError:
            with open(os.path.join('..', 'mydataset', 'myembeddings.pickle'), 'wb') as handle:
                pickle.dump(my_embeddings, handle)

    def load_model(self):
        self.pretrained_model = Model(**self.args)

    def run(self, movie_df, rating_df):
        self.load_model()
        tmp_df = self.refine_behavioral_data(rating_df, self.recommend_limit)
        tmp_df.to_csv(os.path.join(self.path, 'mydataset', 'mapping_df.csv'), mode='w', encoding='utf-8-sig')
        tmp_dic = dict([(a, b) for a, b in zip(tmp_df['book_id'], tmp_df['movieId'])])
        tmp_dic = dict(sorted(tmp_dic.items()))
        self.generate_save_embeddings(tmp_dic, movie_df)
