import os
import asyncio
import psycopg2


class DatabaseConnectionException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class MissingQueryParameterException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def init_db_connection():
    if not os.path.exists('./.env'):
        raise IOError('.env file doesn\'t exist. Can\'t initialize connection to the database!')

    # setting environment variables from .env file
    with open('./.env', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            key, value = line.split('=')
            os.environ[key] = value

    try:
        db_conn = psycopg2.connect(
            user=os.environ['POSTGRES_USERNAME'],
            password=os.environ['POSTGRES_PASSWORD'],
            host=os.environ['POSTGRES_HOST'],
            port=os.environ['POSTGRES_PORT'],
            database=os.environ['POSTGRES_DB']
        )

        return db_conn
    except Exception as e:
        raise DatabaseConnectionException(str(e))


def init_tables():
    try:
        connection = init_db_connection()

        cursor = connection.cursor()
        # read the queries from the .sql file
        with open('./modules/db/sql/create_tables.sql', 'r') as f:
            cursor.execute(f.read())

        connection.commit()
        cursor.close()
        connection.close()
    except DatabaseConnectionException as e:
        print(f'{e} - Can\'t connect to the database! No tables will be created...')

def execute_query(db_conn, cursor, query):
    cursor.execute(query)
    result = cursor.fetchall()[0]

    db_conn.commit()
    cursor.close()
    db_conn.close()

    return result


def report_optuna_study(study, db_conn):
    study_distributions = study.get_trials(deepcopy=False)[0].distributions

    encoded_dim_min = study_distributions['encoded_dim'].low
    encoded_dim_max = study_distributions['encoded_dim'].high
    initial_out_channels_min = study_distributions['initial_out_channels'].low
    initial_out_channels_max = study_distributions['initial_out_channels'].high
    learning_rate_min = study_distributions['lr'].low
    learning_rate_max = study_distributions['lr'].high
    weight_decay_min = study_distributions['weight_decay'].low
    weight_decay_max = study_distributions['weight_decay'].high
    momentum_min = study_distributions['momentum'].low
    momentum_max = study_distributions['momentum'].high
    dampening_min = study_distributions['dampening'].low
    dampening_max = study_distributions['dampening'].high
    scheduler_gamma_min = study_distributions['scheduler_gamma'].low
    scheduler_gamma_max = study_distributions['scheduler_gamma'].high
    kl_divergence_lambda_min = study_distributions['kl_divergence_lambda'].low
    kl_divergence_lambda_max = study_distributions['kl_divergence_lambda'].high
    epochs_min = study_distributions['epochs'].low
    epochs_max = study_distributions['epochs'].high
    batch_size_min = study_distributions['batch_size'].low
    batch_size_max = study_distributions['batch_size'].high
    beta1_min = study_distributions['beta1'].low   
    beta1_max = study_distributions['beta1'].high
    beta2_min =study_distributions['beta2'].low
    beta2_max = study_distributions['beta2'].high
    optimizer_idx_min = study_distributions['optimizer_idx'].low
    optimizer_idx_max = study_distributions['optimizer_idx'].high
    relu_slope_min = study_distributions['relu_slope'].low
    relu_slope_max = study_distributions['relu_slope'].high

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
        momentum_min,
        momentum_max,
        dampening_min,
        dampening_max,
        scheduler_gamma_min,
        scheduler_gamma_max,
        kl_divergence_lambda_min,
        kl_divergence_lambda_max,
        epochs_min,
        epochs_max,
        batch_size_min,
        batch_size_max,
        beta1_min,
        beta1_max,
        beta2_min,
        beta2_max,
        optimizer_idx_min,
        optimizer_idx_max,
        relu_slope_min,
        relu_slope_max
    ) VALUES(
        {encoded_dim_min},
        {encoded_dim_max},
        {initial_out_channels_min},
        {initial_out_channels_max},
        {learning_rate_min},
        {learning_rate_max},
        {weight_decay_min},
        {weight_decay_max},
        {momentum_min},
        {momentum_max},
        {dampening_min},
        {dampening_max},
        {scheduler_gamma_min},
        {scheduler_gamma_max},
        {kl_divergence_lambda_min},
        {kl_divergence_lambda_max},
        {epochs_min},
        {epochs_max},
        {batch_size_min},
        {batch_size_max},
        {beta1_min},
        {beta1_max},
        {beta2_min},
        {beta2_max},
        {optimizer_idx_min},
        {optimizer_idx_max},
        {relu_slope_min},
        {relu_slope_max}
    );
    '''

    cursor = db_conn.cursor()
    cursor.execute(insert_study_query)

    db_conn.commit()
    cursor.close()

def report_optuna_trial(study, trial):
    db_conn = init_db_connection()

    cursor = db_conn.cursor()
    # to get study_id we need to first insert the study into the db and then get the latest study record and extract the id
    # because the study_id is auto generated incrementally
    if trial.number == 0:
        report_optuna_study(study, db_conn)

    # get latest study id
    cursor.execute('''
        SELECT study_id
        FROM optuna_study
        ORDER BY study_id DESC
        LIMIT 1
    ''')
    study_id = cursor.fetchall()[0][0]

    trial_id = trial.number
    trial_state = "'COMPLETED'" if trial.state == 1 else "'PRUNED'"
    trial_encoded_dim = trial.params['encoded_dim']
    trial_initial_out_channels = trial.params['initial_out_channels']
    trial_learning_rate = trial.params['lr']
    trial_weight_decay = trial.params['weight_decay']
    trial_beta1 = trial.params['beta1']
    trial_beta2 = trial.params['beta2']
    trial_momentum = trial.params['momentum']
    trial_dampening = trial.params['dampening']
    trial_optimizer_idx = trial.params['optimizer_idx']
    trial_scheduler_gamma = trial.params['scheduler_gamma']
    trial_kl_divergence_lambda = trial.params['kl_divergence_lambda']
    trial_epochs = trial.params['epochs']
    trial_batch_size = trial.params['batch_size']
    trial_loss_function_id = trial.params['loss_idx']
    trial_relu_slope = trial.params['relu_slope']
    
    if trial.values:
        # if a single objective optimization function
        if len(trial.values) == 1:
            trial_overall_loss_value = trial.values[0]
            trial_kl_divergence_loss_value = -1.0
            trial_loss_value = -1.0
        # multi objective
        else:
            trial_kl_divergence_loss_value = trial.values[0] if trial.values is not None else -1.0
            trial_loss_value = trial.values[1] if trial.values is not None else -1.0
        if trial_kl_divergence_loss_value != -1.0 and trial_loss_value != -1.0:
            trial_overall_loss_value = trial_loss_value + trial_kl_divergence_lambda * trial_kl_divergence_loss_value
    # default values (NOT NULL)
    else:
        trial_overall_loss_value = -1.0
        trial_kl_divergence_loss_value = -1.0
        trial_loss_value = -1.0

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
        {trial_id},
        {trial_state},
        {trial_encoded_dim},
        {trial_initial_out_channels},
        {trial_learning_rate},
        {trial_weight_decay},
        {trial_beta1},
        {trial_beta2},
        {trial_momentum},
        {trial_dampening},
        {trial_optimizer_idx},
        {trial_scheduler_gamma},
        {trial_kl_divergence_lambda},
        {trial_epochs},
        {trial_batch_size},
        {trial_loss_function_id},
        {trial_relu_slope},
        {trial_overall_loss_value},
        {trial_kl_divergence_loss_value},
        {trial_loss_value}
    );
    '''

    cursor.execute(insert_query)

    db_conn.commit()
    cursor.close()
    db_conn.close()

def report_study_best_loss_value(dataset_size):
    db_conn = init_db_connection()
    cursor = db_conn.cursor()

    get_min_loss_value_in_last_study_query = '''
    SELECT study_id, overall_loss_value, kl_divergence_loss_value, loss_value
    FROM optuna_trial AS outr
    WHERE overall_loss_value != -1
    AND study_id = (SELECT MAX(inr.study_id)
                    FROM optuna_study AS inr)
    AND overall_loss_value = (SELECT MIN(inr.overall_loss_value)
                                FROM optuna_trial AS inr
                                WHERE inr.study_id = outr.study_id
                                AND inr.overall_loss_value != -1)
    '''

    cursor.execute(get_min_loss_value_in_last_study_query)
    study_id, best_overall_loss_value, best_kl_divergence_loss_value, best_loss_value = cursor.fetchall()[0]

    update_best_loss_value_query = f'''
        UPDATE optuna_study
        SET best_overall_loss_value = {best_overall_loss_value},
            best_loss_value = {best_loss_value},
            best_kl_divergence_loss_value = {best_kl_divergence_loss_value},
            dataset_size = {dataset_size}
        WHERE study_id = {study_id};
    '''
    cursor.execute(update_best_loss_value_query)

    db_conn.commit()
    cursor.close()
    db_conn.close()

    return study_id

def get_best_hyperparameters(from_last_study=False):
    db_conn = init_db_connection()
    cursor = db_conn.cursor()

    if from_last_study:
        condition = '''
        study_id = (SELECT MAX(inr.study_id)
                    FROM optuna_study AS inr)
        AND overall_loss_value = (SELECT MIN(inr.overall_loss_value)
                                    FROM optuna_trial AS inr
                                    WHERE inr.study_id = outr.study_id
                                    AND inr.overall_loss_value != -1)
        '''
    else:
        condition = '''
            overall_loss_value = (
                SELECT MIN(inr.overall_loss_value)
                FROM optuna_trial AS inr
                WHERE inr.overall_loss_value != -1 AND
                    inr.state != 'PRUNED' AND
                    inr.loss_function_id = 0
            )
        '''

    select_best_hyperparameters_query = f'''
        SELECT encoded_dim, initial_out_channels, learning_rate, weight_decay, beta1, beta2, momentum,
                dampening, optimizer_idx, scheduler_gamma, kl_divergence_lambda, epochs, batch_size, relu_slope
        FROM optuna_trial AS outr
        WHERE {condition};
    '''
    cursor.execute(select_best_hyperparameters_query)
    (encoded_dim, initial_out_channels, learning_rate, weight_decay, beta1, beta2, momentum,
     dampening, optimizer_idx, scheduler_gamma, kl_divergence_lambda, epochs, batch_size, relu_slope) = cursor.fetchall()[0]

    db_conn.commit()
    cursor.close()
    db_conn.close()

    return (encoded_dim, initial_out_channels, learning_rate, weight_decay, (beta1, beta2), momentum,
            dampening, optimizer_idx, scheduler_gamma, kl_divergence_lambda, epochs, batch_size, relu_slope)

def get_last_study_id():
    db_conn = init_db_connection()
    cursor = db_conn.cursor()

    query = 'SELECT MAX(study_id) FROM optuna_study;'

    cursor.execute(query)
    result = cursor.fetchall()[0][0]
    
    db_conn.commit()
    cursor.close()
    db_conn.close()

    return result
