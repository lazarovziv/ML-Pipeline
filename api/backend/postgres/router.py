from typing import Union

from fastapi import APIRouter

from .models.models import TrialRequest, StudyRequest
from .service import PostgresService

class PostgresController:
    def __init__(self):
        self.router = APIRouter(prefix='/optuna')

router = APIRouter(prefix='/optuna')

service = PostgresService()

@router.post('/study/new')
async def report_new_study(item: StudyRequest):
    return await service.report_new_study(study=item)

@router.post('/study/{study_id}/trial/{trial_id}')
async def report_trial_by_id(study_id: int, trial_id: int, item: TrialRequest):
    return await service.report_trial_by_id(study_id=study_id, trial_id=trial_id, trial=item, latest_study=False)

@router.post('/study/latest/trial/{trial_id}')
async def report_trial_to_last_study(trial_id: int, item: TrialRequest):
    return await service.report_trial_to_last_study(trial_id=trial_id, trial=item)

@router.get('/study/latest')
async def get_latest_study():
    return await service.get_latest_study()

@router.get('/study/latest/best_trial')
async def get_best_trial_from_latest_study():
    return await service.get_best_trial_from_latest_study()

@router.get('/study/latest/best_trials/{n}')
async def get_best_n_trials_from_latest_study(n: int):
    return await service.get_best_n_trials_from_latest_study(n=n)

@router.get('/best_hyperparameters')
async def get_best_hyperparameters():
    return await service.get_best_hyperparameters()