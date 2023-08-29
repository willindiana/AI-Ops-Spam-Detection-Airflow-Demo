from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

from spam_detection import helper
from spam_detection import model_1
from spam_detection import model_2
from spam_detection import model_3

from datetime import datetime, timedelta

default_args = {
    'owner': 'will',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}


def greet(ti):
    first_name = ti.xcom_pull(task_ids='get_name', key='first_name')
    last_name = ti.xcom_pull(task_ids='get_name', key='last_name')
    age = ti.xcom_pull(task_ids='get_age', key='age')
    print(f"Hello world my name is {first_name} {last_name} and I am {age}")


def get_name(ti):
    ti.xcom_push(key='first_name', value='Jerry')
    ti.xcom_push(key='last_name', value='Jarvis')


def get_age(ti):
    ti.xcom_push(key='age', value=29)


import datetime
import json
from airflow.decorators import dag, task


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
)
def spam_sms_dag_v3():
    @task()
    def download_dataset():
        print("Download Dataset")
        helper.download_dataset()

    @task()
    def train_model_1():
        print("TODO train_model_1 ")

    @task()
    def train_model_2():
        print("TODO train_model_2 ")

    @task()
    def train_model_3():
        print("TODO train_model_3 ")

    @task()
    def compare_models():
        print("Todod Compare models")

    @task()
    def deploy_winner():
        print("todo Deploy winner")

    download_dataset() >> [train_model_1(), train_model_2(), train_model_3()] >> compare_models() >> deploy_winner()

spam_sms_dag_v3()


# Aiflow DAG stages:
# 1. Download dataset
# 2. Train each of the three models in parallel
# 3. Compare the models
# 4. Deploy the winner
