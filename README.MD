

## Gameplan

## Task List

1. Find AI project and run it - DONE
2. Automate data gathering
3. automate training model
4. automate storing the model
5. automate deploying the model

Aiflow DAG stages:
1. Download dataset
2. Train each of the three models in parallel
3. Compare the models
4. Deploy the winner


Steps to run 
1. export AIRFLOW_HOME=/home/will/Projects/AIOpsDemo
2. Run `airflow standalone`


# Model Performance Expectations
Model Comparison Total Results:
                             accuracy  precision    recall  f1-score
MultinomialNB Model          0.962332   1.000000  0.720000  0.837209
Custom-Vec-Embedding Model   0.979372   0.984733  0.860000  0.918149
Bidirectional-LSTM Model     0.984753   0.985401  0.900000  0.940767
USE-Transfer learning Model  0.982960   0.958042  0.913333  0.935154