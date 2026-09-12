"""Modest, reproducible model-family benchmarks and chronological searches."""
import importlib.util
import logging
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from src.config import CATEGORICAL, INPUT_COLUMNS, SEED
from src.features import make_preprocessor
from src.estimators import PriceRegressor, NativeCatRegressor
from src.evaluation import metrics, date_cv
from src.schema import validate_training

LOGGER = logging.getLogger(__name__)
FORMULATIONS = ['direct', 'log', 'per_sqm']


def model_families() -> list[str]:
    families = ['dummy', 'ridge', 'random_forest', 'hist_gradient_boosting', 'catboost']
    if importlib.util.find_spec('xgboost'):
        families.append('xgboost')
    elif importlib.util.find_spec('lightgbm'):
        families.append('lightgbm')
    return families


def make_model(family: str, formulation: str = 'direct', parameters: dict | None = None,
               include_log_area: bool = True, include_project: bool = True) -> PriceRegressor:
    """One factory for CV, refitting, inference and feature ablations."""
    if family == 'dummy':
        estimator = DummyRegressor(strategy='median')
    elif family == 'ridge':
        estimator = Ridge(alpha=100)
    elif family == 'random_forest':
        estimator = RandomForestRegressor(n_estimators=80, max_depth=18, min_samples_leaf=5,
                                          max_features=.8, n_jobs=2, random_state=SEED)
    elif family == 'hist_gradient_boosting':
        estimator = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=31,
                                                  l2_regularization=10, early_stopping=False, random_state=SEED)
    elif family == 'catboost':
        cats = [c for c in CATEGORICAL if include_project or c != 'PROJECT_EN']
        estimator = NativeCatRegressor(iterations=350, depth=7, learning_rate=.07,
                                      l2_leaf_reg=5, loss_function='RMSE', cat_features=cats,
                                      random_seed=SEED, thread_count=2, has_time=True)
    elif family == 'xgboost':
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=.05,
                                 tree_method='hist', n_jobs=2, random_state=SEED)
    elif family == 'lightgbm':
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(n_estimators=250, num_leaves=31, learning_rate=.05,
                                  n_jobs=2, random_state=SEED, verbosity=-1)
    else:
        raise ValueError(f'Unknown model family: {family}')
    if parameters:
        estimator.set_params(**parameters)
    pipeline = Pipeline([('preprocess', make_preprocessor(family == 'catboost', include_log_area, include_project)),
                         ('regressor', estimator)])
    return PriceRegressor(pipeline, formulation)


def fit_model(name: str, train: pd.DataFrame, formulation: str = 'direct') -> PriceRegressor:
    """Validate a complete labeled table before separating predictors and target."""
    x, y = validate_training(train)
    return make_model(name, formulation).fit(x, y)


def benchmark(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[list[dict], dict]:
    records, fitted = [], {}
    for family in model_families():
        for form in FORMULATIONS:
            key = f'{family}__{form}'
            LOGGER.info('Benchmark %s (%s train rows)', key, len(train))
            start = time.monotonic()
            model = fit_model(family, train, form)
            records.append({'candidate': key, 'model': family, 'formulation': form, 'stage': 'benchmark',
                            'parameters': {}, 'fit_seconds': time.monotonic()-start,
                            **metrics(validation.TRANS_VALUE, model.predict(validation[INPUT_COLUMNS]))})
            fitted[key] = model
    return records, fitted


def tune(family: str, form: str, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, PriceRegressor, pd.DataFrame]:
    """Three draws, two expanding folds, all preceding external validation."""
    spaces = {
        'catboost': {'iterations': [350, 650], 'depth': [6, 8], 'learning_rate': [.04, .08],
                     'l2_leaf_reg': [3, 10], 'loss_function': ['RMSE', 'MAE']},
        'hist_gradient_boosting': {'max_iter': [200, 350], 'max_leaf_nodes': [15, 31, 63],
                                   'l2_regularization': [5, 20], 'learning_rate': [.05, .1],
                                   'loss': ['squared_error', 'absolute_error']},
        'random_forest': {'n_estimators': [80, 140], 'min_samples_leaf': [2, 5, 10],
                          'max_depth': [16, 24], 'max_features': [.7, 1.]},
        'ridge': {'alpha': [1, 10, 100, 1000]},
        'xgboost': {'max_depth': [4, 6, 8], 'n_estimators': [200, 400]},
        'lightgbm': {'num_leaves': [15, 31, 63], 'n_estimators': [200, 400]},
    }
    search = RandomizedSearchCV(make_model(family, form),
        {f'pipeline__regressor__{k}': v for k, v in spaces[family].items()},
        n_iter=3, scoring={'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'},
        refit='MAE', cv=date_cv(train), n_jobs=1, random_state=SEED, error_score='raise')
    LOGGER.info('Tuning %s / %s with 3 configurations × 2 chronological folds', family, form)
    start = time.monotonic()
    search.fit(train[INPUT_COLUMNS], train.TRANS_VALUE)
    params = {k.removeprefix('pipeline__regressor__'): v for k, v in search.best_params_.items()}
    record = {'candidate': f'{family}__{form}__tuned', 'model': family, 'formulation': form,
              'stage': 'tuned', 'parameters': params, 'fit_seconds': time.monotonic()-start,
              **metrics(validation.TRANS_VALUE, search.predict(validation[INPUT_COLUMNS]))}
    return record, search.best_estimator_, pd.DataFrame(search.cv_results_)
