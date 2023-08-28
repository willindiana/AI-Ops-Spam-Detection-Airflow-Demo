

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