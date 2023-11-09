import pendulum
from datetime import timedelta
import time

import pandas as pd
import pendulum
from spam_detection import helper
from spam_detection import model_1
from spam_detection import model_2
from spam_detection import model_3

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
    def data_validation():
        print("TODO Validate Dataset")
        # Could use CheckOperator, ValueCheckOperator or IntervalCheckOperator
        time.sleep(2)

    @task(multiple_outputs=True)
    def train_model_1():
        print("train_model_1 ")
        baseline_model, text_vec, embedding_layer, x_train, y_train, x_test, y_test, x_test_vec = helper.pre_training()
        model_1_instance = model_1.build(text_vec, embedding_layer)
        # Summary of the model 1
        model_1_instance.summary()
        history_1 = model_1_instance.fit(x_train, y_train, 5, validation_data=(x_test, y_test),
                                         validation_steps=int(0.2 * len(x_test)))
        pd.DataFrame(history_1.history).plot()

        model_1_results = helper.evaluate_model(model_1_instance, x_test, y_test)
        print(model_1_results)
        return model_1_results

    @task()
    def train_model_2():
        print("train_model_2 ")
        baseline_model, text_vec, embedding_layer, x_train, y_train, x_test, y_test, x_test_vec = helper.pre_training()
        model_2_instance = model_2.build(text_vec, embedding_layer)
        # Summary of the model 2
        model_2_instance.summary()
        history_2 = helper.fit_model(model_2_instance, 5, x_train, y_train, x_test, y_test)  # fit the model
        pd.DataFrame(history_2.history).plot()

        model_2_results = helper.evaluate_model(model_2_instance, x_test, y_test)
        print(model_2_results)
        return model_2_results

    @task()
    def train_model_3():
        print("train_model_3 ")
        baseline_model, text_vec, embedding_layer, x_train, y_train, x_test, y_test, x_test_vec = helper.pre_training()
        model_3_instance = model_3.build()
        # Summary of model 2
        model_3_instance.summary()
        history_3 = helper.fit_model(model_3_instance, 5, x_train, y_train, x_test, y_test)
        pd.DataFrame(history_3.history).plot()

        model_3_results = helper.evaluate_model(model_3_instance, x_test, y_test)
        print(model_3_results)
        return model_3_results

    @task(provide_context=True)
    def compare_models(**context):
        print("Todo Compare models")
        baseline_model, text_vec, embedding_layer, x_train, y_train, x_test, y_test, x_test_vec = helper.pre_training()
        baseline_model_results = helper.evaluate_model(baseline_model, x_test_vec, y_test)
        model_1_results = context['ti'].xcom_pull(task_ids='train_model_1')
        model_2_results = context['ti'].xcom_pull(task_ids='train_model_2')
        model_3_results = context['ti'].xcom_pull(task_ids='train_model_3')

        total_results = pd.DataFrame({'MultinomialNB Model': baseline_model_results,
                                      'Custom-Vec-Embedding Model': model_1_results,
                                      'Bidirectional-LSTM Model': model_2_results,
                                      'USE-Transfer learning Model': model_3_results}).transpose()

        winner = total_results[total_results['accuracy'] == total_results['accuracy'].max()]

        print(f"The winner is {winner.axes[0].tolist()[0]}")

    @task()
    def validate_winning_model():
        print("Todo Validate Winning model")

    @task()
    def deploy_winner():
        print("todo Deploy winner")

    (download_dataset() >> data_validation() >> [train_model_1(), train_model_2(), train_model_3()] >> compare_models()
     >> validate_winning_model() >> deploy_winner())


spam_sms_dag_v3()
