from airflow.notifications.basenotifier import BaseNotifier

class ExampleNotifier(BaseNotifier):
    def __init__(self, message):
        self.message = message

    # context contains data about the current task
    def notify(self, context):
        title = f'Task {context['task_instance'].task_id} failed!'
        print(title, self.message)