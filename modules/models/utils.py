import os
import torch

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
