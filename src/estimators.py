"""Serializable estimators: target formulations and segment routing in one artifact."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from src.schema import validate_input


class PriceRegressor(RegressorMixin, BaseEstimator):
    """Always predict total AED, regardless of the internal training target.

    The supplied y is total price; no target-derived column enters X. Nonpositive
    regression outputs are floored at AED 1 consistently for every estimator.
    Log predictions are inverted without test-derived smearing corrections.
    """
    def __init__(self, pipeline, formulation: str = 'direct'):
        self.pipeline = pipeline
        self.formulation = formulation

    def fit(self, X: pd.DataFrame, y):
        x = validate_input(X)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(x) or not (np.isfinite(target) & (target > 0)).all():
            raise ValueError('Training target must be one-dimensional, finite, positive and match input rows.')
        if self.formulation == 'direct':
            fitted_target = target
        elif self.formulation == 'log':
            fitted_target = np.log1p(target)
        elif self.formulation == 'per_sqm':
            fitted_target = target / x.ACTUAL_AREA.to_numpy()
        else:
            raise ValueError(f'Unknown target formulation: {self.formulation}')
        if not np.isfinite(fitted_target).all():
            raise ValueError('Target transformation produced nonfinite values.')
        self.pipeline_ = clone(self.pipeline).fit(x, fitted_target)
        self.feature_names_in_ = np.array(x.columns, dtype=object)
        self.n_features_in_ = len(x.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, 'pipeline_')
        x = validate_input(X)
        prediction = np.asarray(self.pipeline_.predict(x), dtype=float).reshape(-1)
        if self.formulation == 'log':
            with np.errstate(over='ignore', invalid='ignore'):
                prediction = np.expm1(prediction)
        elif self.formulation == 'per_sqm':
            prediction = prediction * x.ACTUAL_AREA.to_numpy()
        if len(prediction) != len(x) or not np.isfinite(prediction).all():
            raise ValueError('Model produced nonfinite or incorrectly shaped AED predictions.')
        return np.maximum(prediction, 1.0)


class SegmentedPriceModel(RegressorMixin, BaseEstimator):
    """Fitted global fallback plus optional exact-Flat and exact-Villa specialists."""
    def __init__(self, global_model, specialists=None):
        self.global_model = global_model
        self.specialists = specialists

    def fit(self, X: pd.DataFrame, y):
        """Fit fresh templates for sklearn compatibility; preserve fixed routing."""
        x = validate_input(X)
        target = np.asarray(y, dtype=float)
        self.global_model = clone(self.global_model).fit(x, target)
        fitted = {}
        for segment, template in (self.specialists or {}).items():
            mask = x.PROP_SB_TYPE_EN.eq(segment).to_numpy()
            if not mask.any():
                raise ValueError(f'No training rows for configured specialist: {segment}')
            fitted[segment] = clone(template).fit(x.loc[mask], target[mask])
        self.specialists = fitted
        self.feature_names_in_ = np.array(x.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = validate_input(X)
        prediction = self.global_model.predict(x)
        for segment, model in (self.specialists or {}).items():
            mask = x.PROP_SB_TYPE_EN.eq(segment).to_numpy()
            if mask.any():
                prediction[mask] = model.predict(x.loc[mask])
        return prediction


class NativeCatRegressor(RegressorMixin, BaseEstimator):
    """Clone-safe sklearn adapter for CatBoost's categorical constructor.

    CatBoost 1.2.10 copies cat_features in its constructor, conflicting with
    sklearn 1.9 strict clone checks. Store parameters unchanged here and create
    the native estimator only inside fit. No preprocessing is duplicated.
    """
    def __init__(self, iterations=350, depth=7, learning_rate=.07, l2_leaf_reg=5,
                 loss_function='RMSE', cat_features=None, random_seed=42,
                 thread_count=2, has_time=True):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.cat_features = cat_features
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.has_time = has_time

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(**self.get_params(), verbose=False, allow_writing_files=False)
        self.model_.fit(X, y)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict(X)
