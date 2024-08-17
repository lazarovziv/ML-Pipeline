from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from airflow.decorators import task, dag

import os
import requests
import datetime
import pendulum

@dag(
    dag_id='process_employees',
    schedule_interval='0 0 * * *',
    start_date=pendulum.datetime(2021, 1, 1, tz='UTC'),
    catchup=True,
    dagrun_timeout=datetime.timedelta(minutes=60),
    tags=['employees_table', 'employees']
)

def ProcessEmployees():

    create_employee_table = PostgresOperator(
        task_id='create_employees_table',
        postgres_conn_id='tutorial_pg_conn',
        # can also specify path to .sql file
        sql='''
        CREATE TABLE IF NOT EXISTS employees (
            "Serial Number" NUMERIC PRIMARY KEY,
            "Company Name" TEXT,
            "Employee Name" TEXT,
            "Description" TEXT,
            "Leave" Integer
        );
        '''
    )

    create_employee_temp_table = PostgresOperator(
        task_id='create_employees_temp_table',
        postgres_conn_id='tutorial_pg_conn',
        sql='''
            DROP TABLE IF EXISTS employees_temp;
            CREATE TABLE emplpyees_temp (
                "Serial Number" NUMERIC PRIMARY KEY,
                "Company Name" TEXT,
                "Employee Name" TEXT,
                "Description" TEXT,
                "Leave" Integer
            );
        '''
    )

    @task
    def get_data():
        data_path = '/opt/airflow/dags/files/employees.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        url = 'https://raw.githubusercontent.com/apache/airflow/main/docs/apache-airflow/tutorial/pipeline_example.csv'
        try:
            response = requests.request('GET', url)
        except Exception as e:
            print(e)
            return 1

        with open(data_path, 'w') as f:
            file.write(response.text)

        postgres_hook = PostgresHook(postgres_conn_id, 'tutorial_pg_conn')
        connection = postgres_hook.get_conn()
        cursor = connection.cursor()

        with open(data_path, 'r') as f:
            cursor.copy_expert(
                "COPY employees_temp FROM STDIN WITH CSV HEADER DELIMETER AS ',' QUOTE '\"'",
                f
            )
        connection.commit()
        return 0

    @task
    def merge_data():
        query = '''
            INSERT INTO employees
            SELECT *
            FROM (
                SELECT DISTINCT *
                FROM employees_temp
            ) t
            ON CONFLICT ("Serial Number") DO UPDATE
            SET
                "Employee Name" = excluded."Employee Name",
                "Description" = excluded."Description",
                "Leave" = excluded."Leave";
        '''

        try:
            postgres_hook = PostgresHook(postgres_conn_id='tutorial_pg_conn')
            connection = postgres_hook.get_conn()
            cursor = connection.cursor()

            cursor.execute(query)
            connection.commit()

            return 0
        except Exception as e:
            return 1

    [create_employee_table, create_employee_temp_table] >> get_data() >> merge_data()


dag = ProcessEmployees()
