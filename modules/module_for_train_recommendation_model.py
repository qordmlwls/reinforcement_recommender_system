

# from torch._C import float32
from tqdm import tqdm
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'RecNN')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'RecNN')))
import recnn

# recnn.pd.set("modin") # use modin for data loading and processing
from jupyterthemes import jtplot
jtplot.style(theme='grade3')
from IPython.display import clear_output
import torch
import torch.nn as nn

import os
from os.path import dirname, join
import torch_optimizer as optim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use
os.environ["PYTORCH_ENABLE_NPS_FALLBACK"] = "1"
tqdm.pandas()
from torch.utils.tensorboard import SummaryWriter
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

### 학습
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from RecNN.recnn.nn import ChooseREINFORCE

def string_time_to_unix(s):
    return int(time.mktime(datetime.datetime.strptime(s, "%m/%d/%Y, %H:%M:%S").timetuple()))
def prepare_my_dataset(args_mut, kwargs):
    """
    데이터셋을 준비하는 함수
    """
    # get args
    frame_size = kwargs.get('frame_size')
    key_to_id = args_mut.base.key_to_id
    df = args_mut.df

    # df['liked'] = df['liked'].apply(lambda a: (a - 1) * (1 - a) + a) ## normalization is done by value_net
    # df['liked'] = df['liked'].apply(lambda a: for_like_reward(a))
    df['when'] = df['when'].apply(string_time_to_unix)
    df['book_id'] = df['book_id'].apply(key_to_id.get)
    le = LabelEncoder()
    le.fit(df.reader_id)
    df['reader_id'] = le.transform(df['reader_id'])
    df['reader_id'] = df['reader_id'].astype(int)
    users = df[['reader_id', 'book_id']].groupby(['reader_id']).size()
    users = users[users > frame_size].sort_values(ascending=False).index

    # If using modin: pandas groupby is sync and doesnt affect performance
    # if pd.get_type() == "modin": df = df._to_pandas()
    ratings = df.sort_values(by='when').set_index('reader_id').drop('when', axis=1).groupby('reader_id')

    # Groupby user
    user_dict = {}

    def app(x):
        userid = x.index[0]
        user_dict[int(userid)] = {}
        user_dict[int(userid)]['items'] = x['book_id'].values
        user_dict[int(userid)]['ratings'] = x['liked'].values

    ratings.apply(app)

    args_mut.user_dict = user_dict
    args_mut.users = users

    return args_mut, kwargs

def embed_batch(batch, item_embeddings_tensor, *args, **kwargs):
    return recnn.data.batch_contstate_discaction(batch, item_embeddings_tensor,
                                                 frame_size=kwargs['frame_size'], num_items=kwargs['num_items'])
class Beta(nn.Module):
    def __init__(self, input_size, num_items):
        super(Beta, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, num_items),
            nn.Softmax()
        )
        self.optim = optim.RAdam(self.net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state, action):
        predicted_action = self.net(state)

        loss = self.criterion(predicted_action, action.argmax(1))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return predicted_action.detach()

class TrainRecommendationModel:
    def __init__(self, **args):
        self.model_config = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.dir = None
        self.env = None
        self.frame_size = args['frame_size']
        self.batch_size = args['batch_size']
        self.n_epochs = args['n_epochs']
        self.plot_every = args['plot_every']
        self.num_items = args['num_items']

    def set_config(self):
        if os.path.isfile(join(dirname(dirname(__file__)), 'mydataset', 'frame_env.pkl')):
            os.remove(join(dirname(dirname(__file__)), 'mydataset', 'frame_env.pkl'))

        dirs = recnn.data.env.DataPath(
            base=join(dirname(dirname(__file__)), 'mydataset', ''),
            embeddings="myembeddings.pickle",
            ratings="mydf.csv",
            cache="frame_env.pkl",  # cache will generate after you run
            use_cache=True  # generally you want to save env after it runs
        )
        self.env = recnn.data.env.FrameEnv(
            dirs,
            self.frame_size,
            self.batch_size,
            embed_batch=embed_batch,
            prepare_dataset=prepare_my_dataset,  # <- ! pass YOUR function here
            num_items=self.num_items
        )

        test_batch = next(iter(self.env.test_dataloader))
        state, action, reward, next_state, done = recnn.data.get_base_batch(batch=test_batch, device=self.device)
        self.input_size = len(state[0])

        self.model_config = {
            "frame_size": self.frame_size,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "plot_every": self.plot_every,
            "num_items": self.num_items,
            "input_size": self.input_size
        }
        with open(join(dirname(dirname(__file__)), 'models', 'trained_model', 'model_config.json'), "w") as jsonfile:
            json.dump(self.model_config, jsonfile)

    def reinforce_with_corr(self, **model_config):
        """
        추천 리스트 구하는 데 필요한 모델 학습
        policy_net이 추천 리스트 prediction하는 모델임
        """
        beta_net = Beta(self.input_size, self.num_items).to(self.device)
        value_net = recnn.nn.models.Critic(model_config['input_size'], model_config['num_items'], 2048, 54e-2).to(self.device)
        policy_net = recnn.nn.models.DiscreteActor(model_config['input_size'], model_config['num_items'], 2048).to(self.device)

        reinforce = recnn.nn.Reinforce(policy_net, value_net)
        reinforce = reinforce.to(self.device)

        reinforce.writer = SummaryWriter(log_dir=join(dirname(dirname(__file__)), 'test', 'runs', 'no_corr'))
        plotter = recnn.utils.Plotter(reinforce.loss_layout, [['value', 'policy']], )

        def select_action_corr(state, action, writer, step, **kwargs):
            # note here I provide beta_net forward in the arguments
            return reinforce.nets['policy_net']._select_action_with_correction(state, beta_net.forward, action,
                                                                               writer, step)

        reinforce.nets['policy_net'].select_action = select_action_corr
        reinforce.params['reinforce'] = ChooseREINFORCE(ChooseREINFORCE.reinforce_with_correction)

        while True:
            try:  ###@Todo: histogram안그려지는 오류 해결필요

                for epoch in range(model_config['n_epochs']):
                    for batch in tqdm(self.env.train_dataloader):
                        loss = reinforce.update(batch)
                        reinforce.step()
                        if loss:
                            plotter.log_losses(loss)
                        if reinforce._step % model_config['plot_every'] == 0:
                            clear_output(True)
                            print('step', reinforce._step)
                            ## loss 그래프로 확인하고 싶을때만
                            # plotter.plot_loss()
                        if reinforce._step > 1000:  # max iterations

                            return reinforce, policy_net, value_net
                return reinforce, policy_net, value_net
            except Exception as e:
                print(e)
                continue

    def run(self):
        self.set_config()
        reinforce, policy_net, value_net = self.reinforce_with_corr(**self.model_config)
        print('end')
        ##### 학습된 모델 저장, 추천 리스트 prediction에 필요##################################

        torch.save(policy_net.state_dict(), join(dirname(dirname(__file__)), 'models', 'trained_model', 'policy_net.pt'))