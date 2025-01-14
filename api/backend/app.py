import time
import os

from typing import Union

import asyncio

from fastapi import FastAPI
from pydantic import BaseModel

from postgres import router as postgres_router

app = FastAPI()

app.include_router(postgres_router.router)

@app.get('/')
async def home():
    return { 'hello': 'world' }

@app.get('/wait/{seconds}')
async def wait(seconds: int):
    await sleep(seconds)
    return 200

async def sleep(seconds: int):
    asyncio.sleep(seconds)