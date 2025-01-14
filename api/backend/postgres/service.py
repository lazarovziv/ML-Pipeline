import os

from typing import Union

import io
import psycopg2
# for returning a dictionary from the query that'll be returned as a json
from psycopg2.extras import RealDictCursor

import json

from .models.models import TrialRequest, StudyRequest
from .models.exceptions import DatabaseConnectionException, MissingQueryParameterException

class PostgresService():
    def __init__(self):
        # initialize environment variables manually (would come from a k8s secret)
        if not os.path.exists('./.env'):
            print('.env file doesn\'t exist. Can\'t initialize connection to the database!')
        else:
            with open('./.env', 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    key, value = line.split('=')
                    os.environ[key] = value
        
        self.db_user = os.environ['POSTGRES_USERNAME']
        self.db_password = os.environ['POSTGRES_PASSWORD']
        self.db_host = os.environ['POSTGRES_HOST']
        self.db_port = os.environ['POSTGRES_PORT']
        self.db_name = os.environ['POSTGRES_DB']

        try:
            db_conn = self.create_db_connection()
            # read the queries from the .sql file
            with open('./postgres/models/sql/create_tables.sql', 'r') as f:
                init_query = f.read()
        except DatabaseConnectionException as e:
            print(str(e))
            return
        
        # loop = io.get_event_loop()
        # loop.create_task(self.execute_query(init_query, db_conn))

        self.execute_query(init_query, db_conn)
        
    def create_db_connection(self):
        try:
            db_conn = psycopg2.connect(
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                database=self.db_name
            )

            return db_conn
        except Exception as e:
            raise DatabaseConnectionException(str(e))
        
        return db_conn

    def execute_query(self, query, db_conn, return_query_result=False):
        try:
            cursor = db_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            if return_query_result:
                result = cursor.fetchall()
            db_conn.commit()
            cursor.close()
        except Exception as e:
            print(str(e))
            return 500
            
        if return_query_result:
            return 200, result
        return 200

    def report_new_study(self, study: StudyRequest):
        insert_study_query = f'''
        INSERT INTO optuna_study(
            encoded_dim_min,
            encoded_dim_max,
            initial_out_channels_min,
            initial_out_channels_max,
            learning_rate_min,
            learning_rate_max,
            weight_decay_min,
            weight_decay_max,
            beta1_min,
            beta1_max,
            beta2_min,
            beta2_max,
            momentum_min,
            momentum_max,
            dampening_min,
            dampening_max,
            optimizer_idx_min,
            optimizer_idx_max,
            scheduler_gamma_min,
            scheduler_gamma_max,
            kl_divergence_lambda_min,
            kl_divergence_lambda_max,
            epochs_min,
            epochs_max,
            batch_size_min,
            batch_size_max,
            dataset_size,
            relu_slope_min,
            relu_slope_max
        ) VALUES(
            {study.encoded_dim_min},
            {study.encoded_dim_max},
            {study.initial_out_channels_min},
            {study.initial_out_channels_max},
            {study.learning_rate_min},
            {study.learning_rate_max},
            {study.weight_decay_min},
            {study.weight_decay_max},
            {study.beta1_min},
            {study.beta1_max},
            {study.beta2_min},
            {study.beta2_max},
            {study.momentum_min},
            {study.momentum_max},
            {study.dampening_min},
            {study.dampening_max},
            {study.optimizer_idx_min},
            {study.optimizer_idx_max},
            {study.scheduler_gamma_min},
            {study.scheduler_gamma_max},
            {study.kl_divergence_lambda_min},
            {study.kl_divergence_lambda_max},
            {study.epochs_min},
            {study.epochs_max},
            {study.batch_size_min},
            {study.batch_size_max},
            {study.dataset_size},
            {study.relu_slope_min},
            {study.relu_slope_max}
        );
        '''

        # if any of the try blocks fails, we'll return "internal server error" code
        try:
            db_conn = self.create_db_connection()
        except DatabaseConnectionException as e:
            return 500
        
        task = self.execute_query(insert_study_query, db_conn)
        result =  task
        return result

    def report_trial_by_id(self, study_id: int, trial: TrialRequest):
        db_conn = self.create_db_connection()

        insert_query = f'''
        INSERT INTO optuna_trial(
            study_id,
            trial_id,
            state,
            encoded_dim,
            initial_out_channels,
            learning_rate,
            weight_decay,
            beta1,
            beta2,
            momentum,
            dampening,
            optimizer_idx,
            scheduler_gamma,
            kl_divergence_lambda,
            epochs,
            batch_size,
            loss_function,
            relu_slope,
            overall_loss_value,
            kl_divergence_loss_value,
            loss_value
        ) VALUES (
            {study_id},
            {trial.id},
            {trial.state},
            {trial.encoded_dim},
            {trial.initial_out_channels},
            {trial.learning_rate},
            {trial.weight_decay},
            {trial.beta1},
            {trial.beta2},
            {trial.momentum},
            {trial.dampening},
            {trial.optimizer_idx},
            {trial.scheduler_gamma},
            {trial.kl_divergence_lambda},
            {trial.epochs},
            {trial.batch_size},
            {trial.loss_function_id},
            {trial.relu_slope},
            {trial.overall_loss_value},
            {trial.kl_divergence_loss_value},
            {trial.loss_value}
        );
        '''

        return self.execute_query(insert_query, db_conn)
        
    def report_trial_to_last_study(self, trial_id: int, trial: TrialRequest):
        latest_study =  self.get_latest_study()
        latest_study_id = int(latest_study[0]['study_id'])
        return self.report_trial_by_id(study_id=latest_study_id, trial=trial)

    def get_latest_study(self):
        get_latest_study_query = '''
            SELECT * 
            FROM optuna_study
            WHERE study_id = (
                SELECT MAX(inr.study_id)
                FROM optuna_study AS inr
            );
        '''
        # if any of the try blocks fails, we'll return "internal server error" code
        try:
            db_conn = self.create_db_connection()
        except DatabaseConnectionException as e:
            print(str(e))
            return 500
        
        status_code, query_result =  self.execute_query(get_latest_study_query, db_conn, return_query_result=True)
        return query_result

    def get_all_trials_from_last_study(self):
        get_latest_trials_query = '''
            SELECT * 
            FROM optuna_trial
            WHERE study_id = (
                SELECT MAX(inr.study_id)
                FROM optuna_study AS inr
            )
            ORDER BY trial_id DESC;
        '''
        # if any of the try blocks fails, we'll return "internal server error" code
        try:
            db_conn = self.create_db_connection()
        except DatabaseConnectionException as e:
            print(str(e))
            return 500
        
        status_code, query_result =  self.execute_query(get_latest_trials_query, db_conn, return_query_result=True)
        return query_result

    def get_best_trial_from_latest_study(self):
        get_best_trial_from_latest_study_query = '''
            SELECT *
            FROM optuna_trial AS outr
            WHERE overall_loss_value != -1
            AND study_id = (
                SELECT MAX(inr.study_id)
                FROM optuna_study AS inr
            ) AND overall_loss_value = (
                SELECT MIN(inr.overall_loss_value)
                FROM optuna_trial AS inr
                WHERE inr.study_id = outr.study_id
                AND inr.overall_loss_value != -1
            ) AND outr.state = 'COMPLETED';
        '''

        # if any of the try blocks fails, we'll return "internal server error" code
        try:
            db_conn = self.create_db_connection()
        except DatabaseConnectionException as e:
            print(str(e))
            return 500
        
        status_code, query_result =  self.execute_query(get_best_trial_from_latest_study_query, db_conn, return_query_result=True)
        return query_result
    
    def get_best_n_trials_from_latest_study(self, n):
        get_best_n_trials_from_latest_study_query = f'''
            SELECT *
            FROM optuna_trial AS outr
            WHERE overall_loss_value != -1
            AND study_id = (
                SELECT MAX(inr.study_id)
                FROM optuna_study AS inr
            ) AND outr.state = 'COMPLETED'
            ORDER BY outr.overall_loss_value ASC
            LIMIT {n};
        '''

        # if any of the try blocks fails, we'll return "internal server error" code
        try:
            db_conn = self.create_db_connection()
        except DatabaseConnectionException as e:
            print(str(e))
            return 500
        
        status_code, query_result =  self.execute_query(get_best_n_trials_from_latest_study_query, db_conn, return_query_result=True)
        return query_result

    def get_best_hyperparameters(self):
        pass