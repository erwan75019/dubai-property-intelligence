"""Training/inference feature parity with native or bounded categorical encoding."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import CATEGORICAL, INPUT_COLUMNS, NUMERIC, FORBIDDEN
from src.schema import validate_input


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Target-free date and log-surface features, with optional ablation switches."""
    def __init__(self, include_log_area: bool = True, include_project: bool = True):
        self.include_log_area = include_log_area
        self.include_project = include_project

    def fit(self, X: pd.DataFrame, y=None):
        validate_input(X)
        self.feature_names_in_ = np.array(INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(INPUT_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x = validate_input(X)
        x['LOG_AREA'] = np.log(x.ACTUAL_AREA)
        x['YEAR'] = x.INSTANCE_DATE.dt.year
        x['MONTH'] = x.INSTANCE_DATE.dt.month
        x['QUARTER'] = x.INSTANCE_DATE.dt.quarter
        columns = [c for c in NUMERIC + CATEGORICAL
                   if (self.include_log_area or c != 'LOG_AREA')
                   and (self.include_project or c != 'PROJECT_EN')]
        return x[columns]


def make_preprocessor(native: bool = False, include_log_area: bool = True,
                      include_project: bool = True) -> Pipeline:
    """Fit one-hot vocabulary and scaling only inside a training pipeline.

    CatBoost receives strings directly; it learns categorical statistics inside
    each fit, never from a validation or test target supplied separately.
    """
    steps = [('features', FeatureBuilder(include_log_area, include_project))]
    if not native:
        numeric = [c for c in NUMERIC if include_log_area or c != 'LOG_AREA']
        def encoder(limit: int) -> OneHotEncoder:
            return OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20,
                                 max_categories=limit, sparse_output=False)
        transformers = [('numeric', StandardScaler(), numeric),
                        ('low_cardinality', encoder(25), CATEGORICAL[1:5]),
                        ('area', encoder(150), ['AREA_EN'])]
        if include_project:
            transformers.append(('project', encoder(150), ['PROJECT_EN']))
        steps.append(('encode', ColumnTransformer(transformers, remainder='drop')))
    return Pipeline(steps)
