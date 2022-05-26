# openworld semi-supervised learning
- This project is for testing semi-supervised learning of text
- The research process can be seen here: https://joyous-snout-4cc.notion.site/Classification-Classify-Garbage-documents-using-Open-world-Semi-Supervised-Learning-4226a881f8ad46d195a9b36f0de0e0d9
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


- [test] : 
  - test_preprocessing -> execute preprocessing
  - test_train.py -> execute train
  - test_prediction.py -> execute prediction

