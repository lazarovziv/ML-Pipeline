import time
import os

from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from postgres import postgres_controller

app = FastAPI()

app.include_router(postgres_controller.postgres_router)

@app.get('/')
async def home():
    return { 'hello': 'world' }

@app.get('/wait/{seconds}')
async def wait(seconds: int):
    await sleep(seconds)
    return 200

async def sleep(seconds: int):
    time.sleep(seconds)