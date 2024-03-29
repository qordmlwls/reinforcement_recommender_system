# Reinforcement learning recommender system
- This project is for testing Reinforcement learning recommender system
- The research process can be seen here: https://joyous-snout-4cc.notion.site/Reinforcement-Learning-Social-Media-content-recommender-system-cef42815cae04edfa8d4fca95e0d87a1
## Setting
- python v 3.8.5
- window or linux Ubuntu 20.04.4 LTS
```
pip install -r multiGPU_requirement.txt 
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

```
## Structure
- [mydataset]
  - movies.csv -> information of movies(title, genre)
  - ratings.csv -> rating data of each user
  - myembeddings.pickle -> embeddings of movie content(title + genre)
  - mapping_df.csv -> dataframe for mapping real content_id to label encodings
  - prediction_data_example -> dataframe for prediction
  
- [model] 
  - policy_net.pt -> trained model
  - model_config.json -> model config

- [modules]
  - module_for_data_preparation.py -> data preprocessing module
  - module_for_generating_embedding.py -> data preprocessing module(generate embeddings)
  - module_for_real_time_prediction.py -> prediction module

- [output]
  - rank_df.csv -> output of recommendation list from model prediction
  
- [test] : 
  - test_preprocessing -> execute preprocessing
  - test_train.py -> execute train
  - test_prediction.py -> execute prediction

