import os

os.environ['API_URL'] = 'http://10.0.0.2'
os.environ['API_PORT'] = '8000'

URL = os.environ['API_URL']
PORT = os.environ['API_PORT']
FULL_URL = f'{URL}:{PORT}' if PORT else URL