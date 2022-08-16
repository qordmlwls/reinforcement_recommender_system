# from torch._C import float32
from tqdm import tqdm
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..','RecNN')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'RecNN')))
import recnn

# recnn.pd.set("modin") # use modin for data loading and processing
from jupyterthemes import jtplot
jtplot.style(theme='grade3')
from IPython.display import clear_output
import torch
import torch.nn as nn

import os
import torch_optimizer as optim
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use
tqdm.pandas()
from torch.utils.tensorboard import SummaryWriter
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

### 학습
import time
import datetime
from sklearn.preprocessing import LabelEncoder
def string_time_to_unix(s):
    return int(time.mktime(datetime.datetime.strptime(s, "%m/%d/%Y, %H:%M:%S").timetuple()))

class Beta(nn.Module):
    def __init__(self):
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


def reinforce_with_corr():
    """
    추천 리스트 구하는 데 필요한 모델 학습
    policy_net이 추천 리스트 prediction하는 모델임
    """
    beta_net = Beta().to(device)
    value_net = recnn.nn.models.Critic(input_size, num_items, 2048, 54e-2).to(device)
    policy_net = recnn.nn.models.DiscreteActor(input_size, num_items, 2048).to(device)

    reinforce = recnn.nn.Reinforce(policy_net, value_net)
    reinforce = reinforce.to(device)

    reinforce.writer = SummaryWriter(log_dir=os.path.join('runs','no_corr'))
    plotter = recnn.utils.Plotter(reinforce.loss_layout, [['value', 'policy']], )

    from RecNN.recnn.nn import ChooseREINFORCE

    def select_action_corr(state, action, writer, step, **kwargs):
        # note here I provide beta_net forward in the arguments
        return reinforce.nets['policy_net']._select_action_with_correction(state, beta_net.forward, action,
                                                                           writer, step)

    reinforce.nets['policy_net'].select_action = select_action_corr
    reinforce.params['reinforce'] = ChooseREINFORCE(ChooseREINFORCE.reinforce_with_correction)
    while True:
        try:  ###@Todo: histogram안그려지는 오류 해결필요

            for epoch in range(n_epochs):
                for batch in tqdm(env.train_dataloader):
                    loss = reinforce.update(batch)
                    reinforce.step()
                    if loss:
                        plotter.log_losses(loss)
                    if reinforce._step % plot_every == 0:
                        clear_output(True)
                        print('step', reinforce._step)
                        ## loss 그래프로 확인하고 싶을때만
                        # plotter.plot_loss()
                    if reinforce._step > 1000:  # max iterations

                        return reinforce, policy_net, value_net
            return reinforce, policy_net, value_net
        except:
            continue

# def for_like_reward(like):
#     """
#     reward의 scale을 조정하는 함수
#     """
#     if like > 1200:
#         like = 1200
#     if like < 0:
#         like = 0
#     like = (1/120)*like - 5
#     return like

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
                                                 frame_size=frame_size, num_items=num_items)
##내장함수 상 어쩔 수 없이 arg를 전역변수로 설정함
try:
    keyword_meta = pd.read_csv(os.path.join('..', 'mydataset', 'mapping_df.csv'))
except FileNotFoundError:
    keyword_meta = pd.read_csv(os.path.join(os.getcwd(), 'mydataset', 'mapping_df.csv'))

# -- global variable for embed_batch function
frame_size = 10
batch_size = 10
n_epochs = 100
plot_every = 30
num_items = len(keyword_meta['book_id'])  # n items to recommend. Can be adjusted for your vram
# ---

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # if os.path.isfile(os.path.join('..','mydataset', 'frame_env.pkl')):
    #     os.remove(os.path.join('..','mydataset', 'frame_env.pkl')) ##frame_env는 input_size결정, 추천 목록 수가 항상 바뀔 수 있으므로 삭제 후 시작
    # if os.path.isfile(os.path.join(os.getcwd(),'mydataset', 'frame_env.pkl')):
    #     os.remove(os.path.join(os.getcwd(),'mydataset', 'frame_env.pkl')) ##frame_env는 input_size결정, 추천 목록 수가 항상 바뀔 수 있으므로 삭제 후 시작

    try:
        dirs = recnn.data.env.DataPath(
            base=os.path.join(os.getcwd(),'mydataset',''),
            embeddings="myembeddings.pickle",
            ratings="mydf.csv",
            cache="frame_env.pkl", # cache will generate after you run
            use_cache=True # generally you want to save env after it runs
        )
        env = recnn.data.env.FrameEnv(
            dirs,
            frame_size,
            batch_size,
            embed_batch = embed_batch,
            prepare_dataset=prepare_my_dataset # <- ! pass YOUR function here
        )
    except FileNotFoundError:
        dirs = recnn.data.env.DataPath(
            base=os.path.join('..','mydataset',''),
            embeddings="myembeddings.pickle",
            ratings="mydf.csv",
            cache="frame_env.pkl", # cache will generate after you run
            use_cache=True # generally you want to save env after it runs
        )

        env = recnn.data.env.FrameEnv(
            dirs,
            frame_size,
            batch_size,
            embed_batch = embed_batch,
            prepare_dataset=prepare_my_dataset # <- ! pass YOUR function here
        )
    test_batch = next(iter(env.test_dataloader))
    state, action, reward, next_state, done = recnn.data.get_base_batch(test_batch)
    input_size = len(state[0])

    model_config = {
        "frame_size": frame_size,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "plot_every": plot_every,
        "num_items": num_items,
        "input_size": input_size
    }
    try:
        with open(os.path.join(os.getcwd(), 'model', 'model_config.json'), "w") as jsonfile:
            json.dump(model_config, jsonfile)
    except FileNotFoundError:
        with open(os.path.join('..', 'model', 'model_config.json'), "w") as jsonfile:
            json.dump(model_config, jsonfile)
    reinforce, policy_net, value_net = reinforce_with_corr()
    print('end')
    ##### 학습된 모델 저장, 추천 리스트 prediction에 필요##################################
    try:
        torch.save(policy_net.state_dict(), os.path.join(os.getcwd(), 'model', 'policy_net.pt'))
    except FileNotFoundError:
        torch.save(policy_net.state_dict(), os.path.join('..', 'model', 'policy_net.pt'))
    # ### 모델 로드
    # cuda = torch.device('cuda')
    # policy_net2 = recnn.nn.models.DiscreteActor(input_size, num_items, 2048).to(cuda)
    # policy_net2.load_state_dict(torch.load('..\pretrained_tensorflow_model\policy_net.pt'))
    # policy_net2.eval()