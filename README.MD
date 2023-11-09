# SMS Spam Detection Airflow Training DAG

## Summary

The following project demonstrates the use of apache airflow
train three different models for SMS spam detection,
compare their performance, and declare a winner.

The dataset used can be found here
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

The three models trained were taken from this tutorial
https://www.geeksforgeeks.org/sms-spam-detection-using-tensorflow-in-python/

## Gameplan

## Task List

1. Find AI project and run it - DONE
2. Automate data gathering - DONE
3. automate training model - DONE
4. automate storing the model
5. automate deploying the model

Aiflow DAG stages:
1. Download dataset - DONE
2. Train each of the three models in parallel - DONE
3. Compare the models - DONE
4. Declare the winner - DONE


## Steps to run 
```bash
export AIRFLOW_HOME=/home/will/Projects/AIOpsDemo
airflow standalone
```

## Model Performance Expectations
Model Comparison Total Results:

| Model Name                   | accuracy   | precision | recall    | f1-score|
|-----------------------------|-----------|----------|----------|--------|
| MultinomialNB Model          | 0.962332   | 1.000000  | 0.720000  | 0.837209|
| Custom-Vec-Embedding Model   | 0.979372   | 0.984733  | 0.860000  | 0.918149|
| Bidirectional-LSTM Model     | 0.984753   | 0.985401  | 0.900000  | 0.940767|
| USE-Transfer learning Model  | 0.982960   | 0.958042  | 0.913333  | 0.935154|
