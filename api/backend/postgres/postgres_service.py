import os

from typing import Union

import asyncio
import psycopg2

from .models.models import TrialRequest, StudyRequest
from .models.exceptions import DatabaseConnectionException, MissingQueryParameterException

class PostgresService():
    def __init__(self):
        # initialize environment variables manually (would come from a k8s secret)
        if not os.path.exists('./.env'):
            raise IOError('.env file doesn\'t exist. Can\'t initialize connection to the database!')

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

    async def execute_query(self, query, db_conn, return_query_result=False):
        try:
            cursor = db_conn.cursor()
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

    async def report_new_study(self, study: StudyRequest):
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
        
        return await self.execute_query(insert_study_query, db_conn)

    async def report_trial_by_id(self, study_id: int, trial_id: int, trial: TrialRequest, study: StudyRequest=None, latest_study=True):
        # to get study_id we need to first insert the study into the db and then get the latest study record and extract the id
        # because the study_id is auto generated incrementally
        if trial.id == 0:
            await self.report_new_study(study=study)

        db_conn = self.create_db_connection()
        
        # if we want to report the trial to the latest study' we'll get the latest study id
        if latest_study:
            # get latest study id
            latest_study_id_query = '''
                SELECT study_id
                FROM optuna_study
                ORDER BY study_id DESC
                LIMIT 1
            '''
            return_code, query_result = await self.execute_query(latest_study_id_query, db_conn, return_query_result=True)
            study_id = query_result[0][0]

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
            loss_function_id,
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
        
    async def get_latest_study(self):
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
        
        status_code, query_result = await self.execute_query(get_latest_study_query, db_conn, return_query_result=True)
        return query_result

    async def get_best_trial_from_latest_study(self):
        get_best_trial_from_latest_study_query = '''
            SELECT *
            FROM optuna_trial
            WHERE study_id = (
                SELECT MAX(inr.study_id)
                FROM optuna_study AS inr
            ) 
            AND loss_value = (
                SELECT MIN(inr.loss_value)
                FROM optuna_trial AS inr
            )
            AND state = 'COMPLETED';
        '''

        # if any of the try blocks fails, we'll return "internal server error" code
        try:
            db_conn = self.create_db_connection()
        except DatabaseConnectionException as e:
            print(str(e))
            return 500
        
        status_code, query_result = await self.execute_query(get_best_trial_from_latest_study_query, db_conn, return_query_result=True)
        return query_result