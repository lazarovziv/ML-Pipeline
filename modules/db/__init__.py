import os

# initialize environment variables manually (would come from a k8s secret)
if not os.path.exists('./.env'):
    print('.env file doesn\'t exist. Can\'t initialize connection to the database!')
else:
    with open('./.env', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            # skip comments
            if line.strip().startswith('#'):
                continue
            key, value = line.split('=')
            os.environ[key] = value

FULL_URL = f'{os.environ['URL']}:{os.environ['PORT']}' if os.environ['PORT'] else os.environ['URL']

# os.environ['API_URL'] = 'http://10.0.0.2'
# os.environ['API_PORT'] = '8000'

# URL = os.environ['API_URL']
# PORT = os.environ['API_PORT']
# FULL_URL = f'{URL}:{PORT}' if PORT else URL