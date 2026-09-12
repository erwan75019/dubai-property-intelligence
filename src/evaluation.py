"""Calendar partitions, original-AED metrics and uncertainty diagnostics."""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from src.config import CONFIG, ExperimentConfig


def metrics(y, prediction) -> dict[str, float | None]:
    """MAE/RMSE in AED; undefined single-row or constant-target R² is explicit."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or not len(y):
        raise ValueError('Metrics require matching, nonempty one-dimensional arrays.')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Metrics require finite targets and predictions.')
    return {'MAE': float(mean_absolute_error(y, prediction)),
            'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
            'R2': float(r2_score(y, prediction)) if len(y) > 1 and np.ptp(y) > 0 else None}


def chronological_partitions(df: pd.DataFrame, config: ExperimentConfig = CONFIG) -> dict[str, pd.DataFrame]:
    """Fixed dates, identical for every segment; calibration never fits the model."""
    boundaries = [pd.Timestamp(d) for d in [config.validation_start, config.calibration_start, config.test_start]]
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise ValueError('Split boundaries must be strictly increasing.')
    dates = pd.to_datetime(df.INSTANCE_DATE, errors='raise')
    if dates.isna().any():
        raise ValueError('Missing split dates.')
    a, b, c = boundaries
    masks = {'train': dates < a, 'validation': (dates >= a) & (dates < b),
             'calibration': (dates >= b) & (dates < c), 'test': dates >= c}
    parts = {key: df.loc[mask].sort_values('INSTANCE_DATE').reset_index(drop=True) for key, mask in masks.items()}
    if any(part.empty for part in parts.values()):
        raise ValueError('Every train/validation/calibration/test period must contain records.')
    return parts


def date_cv(df: pd.DataFrame, n_splits: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding whole-day folds, returned as positional indexes for sklearn CV."""
    days = pd.to_datetime(df.INSTANCE_DATE).dt.normalize()
    unique = np.array(sorted(days.unique()))
    if len(unique) < n_splits + 1:
        raise ValueError('Insufficient distinct dates for chronological cross-validation.')
    return [(np.flatnonzero(days.isin(unique[tr])), np.flatnonzero(days.isin(unique[va])))
            for tr, va in TimeSeriesSplit(n_splits=n_splits).split(unique)]


def temporal_split(df: pd.DataFrame, fraction: float = .2):
    """Legacy notebook helper, retaining complete calendar days."""
    days = pd.to_datetime(df.INSTANCE_DATE).dt.normalize()
    dates = sorted(days.unique())
    if len(dates) < 3 or not 0 < fraction < 1:
        raise ValueError('Need three dates and a fraction between zero and one.')
    cutoff = dates[min(len(dates)-1, max(1, int(len(dates)*(1-fraction))))]
    return df.loc[days < cutoff].copy(), df.loc[days >= cutoff].copy()


def calibrate_interval(y, prediction, alpha: float = .1) -> dict:
    """Split-conformal-style multiplicative interval on chronological residuals.

    Time dependence violates exchangeability; nominal coverage is not guaranteed.
    No calibration target is used for estimator selection or fitting.
    """
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if not 0 < alpha < 1 or y.shape != prediction.shape or not len(y):
        raise ValueError('Invalid calibration arrays or alpha.')
    if not (np.isfinite(y) & np.isfinite(prediction) & (y > 0) & (prediction > 0)).all():
        raise ValueError('Calibration requires finite positive values.')
    scores = np.abs(np.log(y) - np.log(prediction))
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError('Calibration sample is too small for the requested coverage.')
    return {'alpha': alpha, 'log_radius': float(np.sort(scores)[rank-1]), 'n': len(scores),
            'method': 'chronological split calibration, absolute log residual',
            'caveat': 'Nominal coverage is not guaranteed under temporal drift or subgroup shift.'}


def interval_bounds(prediction, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    radius = float(calibration['log_radius'])
    if not np.isfinite(radius) or radius < 0 or not (np.isfinite(prediction) & (prediction > 0)).all():
        raise ValueError('Invalid interval inputs.')
    lower, upper = prediction * np.exp(-radius), prediction * np.exp(radius)
    if not np.isfinite(upper).all():
        raise ValueError('Nonfinite interval bounds.')
    return lower, upper


def error_tables(test: pd.DataFrame, prediction, lower, upper) -> dict[str, pd.DataFrame]:
    """Post-selection subgroup diagnostics; price bands are target-based analysis only."""
    data = test.assign(predicted_price=prediction, lower=lower, upper=upper)
    data['price_band'] = pd.cut(data.TRANS_VALUE, [0, 1e6, 2e6, 5e6, 10e6, np.inf],
                                labels=['<1M', '1–2M', '2–5M', '5–10M', '10M+'], right=False)
    tables = {}
    for column in ['PROP_SB_TYPE_EN', 'AREA_EN', 'price_band', 'ROOMS_EN', 'IS_OFFPLAN_EN']:
        rows = []
        for name, group in data.groupby(column, dropna=False, observed=True):
            rows.append({'group': str(name), 'rows': len(group), **metrics(group.TRANS_VALUE, group.predicted_price),
                         'bias_AED': float((group.predicted_price - group.TRANS_VALUE).mean()),
                         'interval_coverage': float(group.TRANS_VALUE.between(group.lower, group.upper).mean())})
        tables[column] = pd.DataFrame(rows).sort_values('rows', ascending=False)
    return tables


def paired_day_bootstrap(test: pd.DataFrame, old, new, seed: int = 42) -> dict:
    """Paired resampling of calendar days, a limited uncertainty diagnostic."""
    data = pd.DataFrame({'day': pd.to_datetime(test.INSTANCE_DATE).dt.normalize(),
                         'difference': np.abs(np.asarray(old) - test.TRANS_VALUE.to_numpy()) - np.abs(np.asarray(new) - test.TRANS_VALUE.to_numpy())})
    day = data.groupby('day').difference.agg(['sum', 'count']).to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1000):
        sampled = day[rng.integers(0, len(day), len(day))]
        values.append(sampled[:, 0].sum() / sampled[:, 1].sum())
    return {'MAE_reduction_AED': float(data.difference.mean()),
            'day_bootstrap_95pct': np.quantile(values, [.025, .975]).tolist(),
            'caveat': 'Days may be dependent; this does not establish robustness in a new market regime.'}
