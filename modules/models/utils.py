import os

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.metrics import recall_score, precision_score, confusion_matrix, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

import torch
import torch.nn as nn

from skimage import feature

import optuna

def create_model_name(opt, params):
    if opt:
        final_name = f'{opt.__class__.__name__}-'
        for key, value in opt.state_dict()['param_groups'][0].items():
            if type(value) == type([]) or \
                    key == 'differentiable' or \
                    key == 'foreach' or \
                    key == 'maximize' or \
                    key == 'fused' or \
                    key == 'nesterov':
                continue
            if key == 'lr':
                final_name += f'lr={params['lr']}-'
                continue
            final_name += f'{key}={value}-'
    else:
        final_name = ''
    final_name += f'lr={params['lr']}'
    final_name += f'kl_lambda={params['kl_divergence_lambda']}-'
    final_name += f'epochs={params['epochs']}-'
    final_name += f'encoded_dim={params['encoded_dim']}-'
    final_name += f'channels={params['initial_out_channels']}-'
    final_name += f'bs={params['batch_size']}'
    return final_name

def save_model(model, opt, params):
    file_name_prefix = f'./saved_models/{create_model_name(opt, params)}'
    file_name = f'{file_name_prefix}.pt'
    if os.path.exists(file_name):
        counter = 0
        file_name = f'{file_name_prefix}-{counter}.pt'
        while os.path.exists(file_name):
            print(f'file {file_name} already exists, creating a new file...')
            counter += 1
            file_name = f'{file_name_prefix}-{counter}.pt'
    print(f'length: {len(file_name)}')
    torch.save(model.state_dict(), file_name)
    print(f'file {file_name} created.')

def model_match_params(file_name, params):
    # params_matched_model_name = create_model_name(opt=None, params=params)
    # return params_matched_model_name in file_name
    encoded_dim = params['encoded_dim']
    initial_out_channels = params['initial_out_channels']
    lr = params['lr']
    weight_decay = params['weight_decay']
    momentum = params['momentum']
    dampening = params['dampening']
    scheduler_gamma = params['scheduler_gamma']
    kl_divergence_lambda = params['kl_divergence_lambda']
    epochs = params['epochs']
    batch_size = params['batch_size']

    return f'encoded_dim={encoded_dim}' in file_name and \
    f'channels={initial_out_channels}' in file_name and \
    f'lr={lr}' in file_name and \
    f'kl_lambda={kl_divergence_lambda}' in file_name and \
    f'bs={batch_size}' in file_name

def load_model(params):
    models_names = os.listdir('./saved_models')
    params_matched_model_names = [name for name in models_names if model_match_params(name, params)]
    # if the model is unique (was trained multiple times by different executions of the notebook)
    if len(params_matched_model_names) == 1:
        return torch.load(f'./saved_models/{params_matched_model_names[0]}', weights_only=True)
    
    suffixes = [name.split('-')[-1] for name in params_matched_model_names]
    model_numbers = [int(suffix.split('.pt')[0]) for suffix in suffixes if suffix.split('.pt')[0].isdigit()]
    latest_model_number = max(model_numbers)
    file_name = list(filter(lambda s: s.endswith(f'{latest_model_number}.pt') and model_match_params(s, params), params_matched_model_names))[0]
    return torch.load(f'./saved_models/{file_name}', weights_only=True)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

def get_loss_function(loss_idx):
    return nn.MSELoss() if loss_idx == 0 else RMSELoss()

# params is a dictionary of type 'param_name': ['variable_type', [value_0, value_1] ]
def initialize_custom_hyperparameters(trial, params):
    initialized_params = {}
    for param_name in params.keys():
        param_type = params[param_name][0]
        param_values = params[param_name][1]
        if param_type == 'float':
            initialized_params[param_name] = trial.suggest_float(param_name, *param_values)
        elif param_type == 'int':
            initialized_params[param_name] = trial.suggest_int(param_name, *param_values)
        elif param_type == 'categorical':
            initialized_params[param_name] = trial.suggest_categorical(param_name, param_values)
    return initialized_params

def show_metrics(y_true, y_preds, classes, plot=True):
    accuracy = (y_preds == y_true).sum() / y_preds.shape[0]

    print(f'Total Accuracy: {accuracy}\n')

    average_policy = None

    # precision = tp / (tp + fp)
    print('Precision: ')
    precisions = precision_score(y_true, y_preds, average=average_policy, zero_division=0)
    for class_idx in range(len(classes)):
        print(f'{classes[class_idx]}: {precisions[class_idx]}')
    print()

    # recall = tp / (tp + fn)
    print('Recall: ')
    recalls = recall_score(y_true, y_preds, average=average_policy, zero_division=0)
    for class_idx in range(len(classes)):
        print(f'{classes[class_idx]}: {recalls[class_idx]}')
    print()

    print('F1 Score: ')
    f1_scores = f1_score(y_true, y_preds, average=average_policy, zero_division=0)
    for class_idx in range(len(classes)):
        print(f'{classes[class_idx]}: {f1_scores[class_idx]}')
    print()

    if plot:
        g = sns.heatmap(confusion_matrix(y_true, y_preds), annot=True, fmt='.3g', yticklabels=classes)

        g.set_xticklabels(labels=[f'{class_name}' for class_name in classes], rotation=60)
        
        plt.xlabel('Prediction')
        plt.ylabel('Truth')
        plt.title(f'Accuracy: {np.sum(y_true == y_preds) / y_true.shape[0]}')

def normalize_images(imgs, lib=np):
    imgs += lib.abs(imgs.min())
    imgs /= imgs.max()
    imgs *= 255
    return imgs.astype(np.uint8)

# if it's the test dataset we need to use the mean and std from the training dataset
def normalize(arr, mean=None, std=None, is_train=True):
    mean = arr.mean() if is_train else mean
    std = arr.std() if is_train else std
    return (arr - mean) / std

def create_dataset(X, y, means=None, stds=None, drop_cols=None, largest_descriptor_dim=None, descriptor_shape=None, boundary_pixels=None, is_train=True):
    # set test dataset dimensions to match the training's
    if not is_train and boundary_pixels is not None:
        uppermost_pixel, bottommost_pixel, rightmost_pixel, leftmost_pixel = boundary_pixels
        X = X[:, uppermost_pixel:bottommost_pixel+1, leftmost_pixel:rightmost_pixel+1]

    df = pd.DataFrame(columns=['id', 'class'])
    df['id'] = np.arange(X.shape[0])
    df['class'] = y

    # calculating gradients
    sobel_x_gradients = sp.ndimage.sobel(X / 255, axis=1)
    sobel_y_gradients = sp.ndimage.sobel(X / 255, axis=2)
    mag, angle = cv2.cartToPolar(sobel_x_gradients, sobel_y_gradients, angleInDegrees=True)
    flattened_magnitudes = mag.flatten().reshape(X.shape[0], -1)
    gradient_stds = np.array([sample.std() for sample in flattened_magnitudes])
    df['gradient_std'] = gradient_stds
    gradients_norma = np.linalg.norm(flattened_magnitudes, axis=1)
    df['gradient_norma'] = gradients_norma
    # HoGs
    df[[f'hog_descriptor_{i}' for i in range(descriptor_shape)]] = np.array([feature.hog(sample, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=True, channel_axis=None)[0] for sample in X])

    # histograms for all images
    orb = cv2.ORB_create()

    descriptors = []
    key_points = []

    for i in range(X.shape[0]):
        image_key_points, image_descriptors = orb.detectAndCompute(X[i], None)
        descriptors.append(image_descriptors)
        key_points.append(cv2.KeyPoint_convert(image_key_points))

    orb_histograms = np.zeros((X.shape[0], largest_descriptor_dim))
    for image_idx in range(len(descriptors)):
        if descriptors[image_idx] is None:
            continue
        for desc_value in descriptors[image_idx]:
            orb_histograms[image_idx, desc_value] += 1
    df[[f'orb_histogram_{i}' for i in range(largest_descriptor_dim)]] = orb_histograms

    # removing zero variance columns only if it's the training dataset. those columns will be used to drop the same columns in the test dataset
    if is_train:
        df_cols_to_drop = []
        df_cols_means = {}
        df_cols_stds = {}

        std_epsilon = 1e-4

        for col in df.columns:
            std = df[col].std()
            if std < std_epsilon:
                df.drop(col, axis=1, inplace=True)
                df_cols_to_drop.append(col)
            else:
                df_cols_means[col] = df[col].mean()
                df_cols_stds[col] = df[col].std()
        means = df_cols_means
        stds = df_cols_stds
    
    if drop_cols is not None:
        df.drop(drop_cols, axis=1, inplace=True)
    
    # normalizing columns since most of our features are normally distributed
    for col in df.drop('class', axis=1).columns:
        df[col] = normalize(df[col], mean=means[col], std=stds[col], is_train=is_train)

    if is_train:
        return df, df_cols_to_drop, df_cols_means, df_cols_stds
    return df

def custom_optimize(study_name, objective, n_trials=50):
    n_trials = np.clip(n_trials, 1, 50)
    timeout_factor = np.clip(n_trials // 30, 1, 20)

    study = optuna.create_study(study_name=study_name, directions=['maximize'])
    # study = optuna.create_study(study_name=study_name, directions=directions)
    study.optimize(objective, n_trials=n_trials, timeout=20000*timeout_factor, show_progress_bar=True, gc_after_trial=True)

    return study

def get_dataset_params():
    params = {}
    with open('./params.txt', 'r') as f:
        for line in f.readlines():
            line_text = line.split('=')
            params[line_text[0]] = int(line_text[1][:-1])
    return params
    