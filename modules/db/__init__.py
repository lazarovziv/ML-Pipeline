import os

# initialize environment variables manually (would come from a k8s secret)
if not os.path.exists(f'{os.path.dirname(os.path.abspath(__file__))}/.env'):
    print('.env file doesn\'t exist. Can\'t initialize connection to the API!')
else:
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/.env', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            # skip comments
            if line.strip().startswith('#'):
                continue
            key, value = line.split('=')
            os.environ[key] = value

FULL_URL = f'{os.environ['API_URL']}:{os.environ['API_PORT']}' if os.environ['API_PORT'] else os.environ['API_URL']
