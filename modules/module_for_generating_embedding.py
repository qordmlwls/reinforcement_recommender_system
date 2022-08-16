from tqdm import tqdm
import pandas as pd
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import emoji
from soynlp.normalizer import repeat_normalize
import time
##koelectra
from torch.nn import functional as F
from transformers import AutoTokenizer, ElectraTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
import unicodedata
args = {
    'random_seed': 42,  # Random Seed
    'pretrained_model': "bert-based-uncased",  # Transformers PLM name
    'pretrained_tokenizer': "bert-based-uncased",
    # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    'batch_size': 32,
    'lr': 5e-6,  # Starting Learning Rate
    'epochs': 5,  # Max Epochs
    'max_length': 150,  # Max Length mydataset size
    'train_data_path': "../mydataset/SNS_content_classification_train.csv",  # Train Dataset file
    'val_data_path': '../mydataset/test_prediction_pl.csv',  # Validation Dataset file
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',  # AdamW vs AdamP
    'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
    'fp16': True,  # Enable train on FP16
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    'cpu_workers': os.cpu_count(),
}
class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        try:
            # self.bert = BertModel.from_pretrained(os.path.abspath(os.path.join(os.getcwd(),'models', 'pretrained_tensorflow_model')),
            #                                       local_files_only=True)
            self.bert = BertModel.from_pretrained(self.hparams.pretrained_model)
            # self.tokenizer = BertTokenizer.from_pretrained(
            #     os.path.abspath(os.path.join(os.getcwd(), 'models', 'pretrained_tensorflow_model')), local_files_only=True)
            self.tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_tokenizer)
        except OSError:
            self.bert = BertModel.from_pretrained(os.path.abspath(os.path.join('..', 'models', 'pretrained_tensorflow_model')), local_files_only=True)
            # self.bert = BertForSequenceClassification.from_pretrained(self.hparams.pretrained_model)
            self.tokenizer = BertTokenizer.from_pretrained(os.path.abspath(os.path.join('..', 'models', 'pretrained_tensorflow_model')), local_files_only=True)
    def forward(self, **kwargs):
        return self.bert(**kwargs)
    def step(self, batch, batch_idx):
        data = batch
        output = self(input_ids=data)
        # Transformers 4.0.0+
        # loss = output.loss
        # logits = output.logits
        # preds = logits.argmax(dim=-1)
        # y_true = list(labels.cpu().numpy())
        # y_pred = list(preds.cpu().numpy())
        return {
            # 'loss': loss,
            # 'y_true': y_true,
            # 'y_pred': y_pred,
            'output' : output
        }
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)
    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)
        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
        self.log(state + '_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state + '_acc', accuracy_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state + '_precision', precision_score(y_true, y_pred, average='micro'), on_epoch=True, prog_bar=True)
        self.log(state + '_recall', recall_score(y_true, y_pred, average='micro'), on_epoch=True, prog_bar=True)
        self.log(state + '_f1', f1_score(y_true, y_pred, average='micro'), on_epoch=True, prog_bar=True)
        return {'loss': loss}
    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='train')
    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')
    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path,  lineterminator='\n')
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, delimiter='\t', header=0, encoding="latin-1")
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')
    def preprocess_dataframe(self, df):
        #         def clean_ascii(text):
        #             # function to remove non-ASCII chars from data
        #             return ''.join(i for i in text if ord(i) < 128)
        # df['Tweet'] = df['Tweet'].apply(clean_ascii)
        # df = pd.concat([df['Target'], df['Tweet']], axis=1)
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        ##Encoding the Labels
        # possible_labels = df.Target.unique()
        # print(df.Target.unique())
        # label_dict = {}
        # for index, possible_label in enumerate(possible_labels):
        #     label_dict[possible_label] = index
        # df['label'] = df.Target.replace(label_dict)
        print(df.head())
        print(df.tail())
        # print(label_dict)
        def rm_emoji(Data):
            return Data.encode('utf-8','ignore').decode('utf-8')
        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.replace('\n','').strip()
            x = unicodedata.normalize('NFC',x)
            x = repeat_normalize(x, num_repeats=2)
            x = rm_emoji(x)
            return x
        df['document'] = df['document'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            padding='max_length',
            max_length=self.hparams.max_length,
            truncation=True,
        ))
        return df
    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df)
        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            # torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.cpu_workers,
        )
    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)
    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, shuffle=False)
def get_embedding(text : str, pretrained_model : None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda")
    pretrained_model.to(device)
    pretrained_model.eval()
    pretrained_model.freeze()
    output = pretrained_model(**pretrained_model.tokenizer(text,return_tensors='pt',padding='max_length',max_length=args['max_length'],truncation=True).to(device)) ## gpu로 올려준다
    embedding = output.pooler_output.squeeze().detach().cpu() ## 임배딩은 1차원의 텐서이어야 함(torch.Tensor), squeeze는 1인 차원 제거해줌
    return embedding
# pretrained_model = Model(**args)



