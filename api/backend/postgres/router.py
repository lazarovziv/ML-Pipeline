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
def report_new_study(item: StudyRequest):
    return service.report_new_study(study=item)

@router.post('/study/{study_id}/trial')
def report_trial_to_study(study_id: int, item: TrialRequest):
    return service.report_trial_by_id(study_id=study_id, trial=item)

@router.post('/study/trial')
def report_trial_to_last_study(item: TrialRequest):
    return service.report_trial_to_last_study(trial_id=item.id, trial=item)

@router.get('/study/latest')
def get_latest_study():
    return service.get_latest_study()

@router.get('/study/latest/all_trials')
def get_all_trials_from_last_study():
    return service.get_all_trials_from_last_study()

@router.get('/study/latest/best_trial')
def get_best_trial_from_latest_study():
    return service.get_best_trial_from_latest_study()

@router.get('/study/latest/best_trials/{n}')
def get_best_n_trials_from_latest_study(n: int):
    return service.get_best_n_trials_from_latest_study(n=n)

@router.get('/best_hyperparameters')
def get_best_hyperparameters():
    return service.get_best_hyperparameters()