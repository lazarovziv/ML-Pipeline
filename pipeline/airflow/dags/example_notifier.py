from datetime import datetime

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

from airflow.notifications.basenotifier import BaseNotifier

class ExampleNotifier(BaseNotifier):
    def __init__(self, message):
        self.message = message

    # context contains data about the current task
    def notify(self, context):
        title = f'Task {context['task_instance'].task_id} failed!'
        print(title, self.message)

with DAG(
    dag_id='example_notifier',
    start_date=datetime(2022, 1, 1),
    schedule_interval=None,
    on_success_callback=ExampleNotifier('Success!'),
    on_failure_callback=ExampleNotifier('Failure!')
) as dag:
    task0 = BashOperator(
        task_id='example_task0',
        bash_command='exit 1',
        on_success_callback=ExampleNotifier('Task Succeeded!')
    )

    task1 = BashOperator(
        task_id='example_task1',
        bash_command='echo Hello $name!',
        env={ 'name': 'Person' }
    )

    task0 >> task1
