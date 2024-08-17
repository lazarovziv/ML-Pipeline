from datetime import datetime

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

from notifiers import ExampleNotifier

with DAG(
    dag_id='example_notifier',
    start_date=datetime(2022, 1, 1),
    schedule_interval=None,
    on_success_callback=ExampleNotifier('Success!'),
    on_failure_callback=ExampleNotifier('Failure!')
):
    task = BashOperator(
        task_id='example_task',
        bash_command='exit 1',
        on_success_callback=ExampleNotifier('Task Succeeded!')
    )
