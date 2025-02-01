# Ziv Lazarov - Data Science Project - The Open University of Israel
## In this project I've been working on solving a classification problem for identifying Alzheimer's in patients and their level of dementia.

All explanations are in the notebooks.

The order of which I recommend to walk through the project is:
* EDA + Augmentation notebook - [link](https://nbviewer.org/github/lazarovziv/ML-Pipeline/blob/main/image-augmentation.ipynb) - some of the plots won't render here on GitHub so I had to find a workaround for which it'll be possible to view outside my workspace, so `nbviewer` is the way to go if you're interested in the interactive plots.
* Fine tuning + Classification - [link](https://nbviewer.org/github/lazarovziv/ML-Pipeline/blob/main/image-classification.ipynb) - same reason.

NOTE: There are modules used in these notebooks that were written in `.py` files to make them a bit cleaner, so if the code interests you, it is here in this repository under the [modules](https://github.com/lazarovziv/ML-Pipeline/tree/main/modules) directory, and for the API, it is [here](https://github.com/lazarovziv/ML-Pipeline/tree/main/api).

For anyone interested on running those workflows manually, please install `docker` before starting to be able to run the API and database.

Before running the following commands, please fill the `.env` files in `modules/db` and in `api/backend/.env` with the relevant values.

```bash
# creating the directories for the tensorboard logs
mkdir logs
mkdir augmentation
mkdir classification
# running the tensorboard in the background
tensorboard --logdir ./logs &

cd api

# create a local network for the api and db to communicate via their respective IPs
docker network create --driver bridge <db-api-network-name>

# build the db image
docker build -t <name-of-db-image>:<db-image-tag> -f postgres.Dockerfile .

# build the fastapi image
docker build -t <name-of-api-image>:<api-image-tag> -f fastapi.Dockerfile .

# if you don't want to persist the DB's data on your computer, don't specify the "-v" flag an argument.
docker run -d -p 5432:5432 --name postgresql --net <db-api-network-name> -e POSTGRES_USER=<same-from-.env> -e POSTGRES_PASSWORD=<same-from-.env> -e POSTGRES_DB=<same-from-.env> -v $(pwd)/postgresql:/var/lib/postgresql/data postgres:latest

# run the api after the db finishes to start
docker run -d -p 8000:80 --name fastapi --net <db-api-network-name> <name-of-api-image>:<image-tag>
```