import os
import asyncio
import requests
import psycopg2

from modules.models.utils import get_loss_function

os.environ['API_URL'] = 'http://10.0.0.2'
os.environ['API_PORT'] = '8001'

URL = os.environ['API_URL']
PORT = os.environ['API_PORT']
FULL_URL = f'{URL}:{PORT}' if PORT else URL

class MissingQueryParameterException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def report_optuna_study(study):
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
    beta2_min = study_distributions['beta2'].low
    beta2_max = study_distributions['beta2'].high
    optimizer_idx_min = study_distributions['optimizer_idx'].low
    optimizer_idx_max = study_distributions['optimizer_idx'].high
    relu_slope_min = study_distributions['relu_slope'].low
    relu_slope_max = study_distributions['relu_slope'].high
    dataset_size = study_distributions['dataset_size'].low

    json_data = {
        'encoded_dim_min': encoded_dim_min,
        'encoded_dim_max': encoded_dim_max,
        'initial_out_channels_min': initial_out_channels_min,
        'initial_out_channels_max': initial_out_channels_max,
        'learning_rate_min': learning_rate_min,
        'learning_rate_max': learning_rate_max,
        'weight_decay_min': weight_decay_min,
        'weight_decay_max': weight_decay_max,
        'momentum_min': momentum_min,
        'momentum_max': momentum_max,
        'dampening_min': dampening_min,
        'dampening_max': dampening_max,
        'scheduler_gamma_min': scheduler_gamma_min,
        'scheduler_gamma_max': scheduler_gamma_max,
        'kl_divergence_lambda_min': kl_divergence_lambda_min,
        'kl_divergence_lambda_max': kl_divergence_lambda_max,
        'epochs_min': epochs_min,
        'epochs_max': epochs_max,
        'batch_size_min': batch_size_min,
        'batch_size_max': batch_size_max,
        'beta1_min': beta1_min,
        'beta1_max': beta1_max,
        'beta2_min': beta2_min,
        'beta2_max': beta2_max,
        'optimizer_idx_min': optimizer_idx_min,
        'optimizer_idx_max': optimizer_idx_max,
        'relu_slope_min': relu_slope_min,
        'relu_slope_max': relu_slope_max,
        'dataset_size': dataset_size # size of the train dataset, without val
    }

    url = f'{FULL_URL}/optuna/study/new'
    response = requests.post(url, json=json_data)

    return response.status_code

def report_optuna_trial(study, trial):
    if trial.number == 0:
        report_optuna_study(study)
    
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
        # single objective function
        if len(trial.values) == 1:
            trial_overall_loss_value = trial.values[0]
            trial_loss_value = -1.0
            trial_kl_divergence_loss_value = -1.0
        else:
            trial_kl_divergence_loss_value = trial.values[0]
            trial_loss_value = trial.values[1]
            trial_overall_loss_value = trial_loss_value + trial_kl_divergence_lambda * trial_kl_divergence_loss_value
    else:
        trial_overall_loss_value, trial_loss_value, trial_kl_divergence_loss_value = -1.0, -1.0, -1.0

    json_data = {
        'id': trial_id,
        'state': trial_state,
        'encoded_dim': trial_encoded_dim,
        'initial_out_channels': trial_initial_out_channels,
        'learning_rate': trial_learning_rate,
        'weight_decay': trial_weight_decay,
        'beta1': trial_beta1,
        'beta2': trial_beta2,
        'momentum': trial_momentum,
        'dampening': trial_dampening,
        'optimizer_idx': trial_optimizer_idx,
        'scheduler_gamma': trial_scheduler_gamma,
        'kl_divergence_lambda': trial_kl_divergence_lambda,
        'epochs': trial_epochs,
        'batch_size': trial_batch_size,
        'loss_function_id': trial_loss_function_id,
        'overall_loss_value': trial_overall_loss_value,
        'kl_divergence_loss_value': trial_kl_divergence_loss_value,
        'loss_value': trial_loss_value,
        'relu_slope': trial_relu_slope
    }

    url = f'{FULL_URL}/optuna/study/trial'
    response = requests.post(url, json=json_data)

    return response.status_code

def get_best_hyperparameters(from_last_study=False):
    url_path = 'latest/best_trial' if from_last_study else 'best_hyperparameters'

    url = f'{FULL_URL}/optuna/study/{url_path}'
    response = requests.get(url)

    return response.json()

def get_last_study_id():
    url_path = 'optuna/study/latest'
    url = f'{FULL_URL}/{url_path}'
    response = requests.get(url)

    return response.json()[0]['study_id']
