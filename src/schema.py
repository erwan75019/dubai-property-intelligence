"""Explicit, order-independent feature and training-data validation."""
import re
from collections.abc import Sequence
import numpy as np
import pandas as pd
from src.config import (CATEGORICAL, FORBIDDEN, INPUT_COLUMNS, MAX_SURFACE,
                        OPTIONAL_COLUMNS, RESIDENTIAL_TYPES, TARGET)


def assert_safe_features(names: Sequence[str]) -> None:
    """An allowlist also blocks unknown target-derived fields, not just known names."""
    if len(names) != len(set(names)):
        raise ValueError('Duplicate feature names are not allowed.')
    unsafe = set(names) & FORBIDDEN
    unknown = set(names) - set(INPUT_COLUMNS)
    if unsafe or unknown:
        raise ValueError(f'Forbidden or unexpected predictor columns: {sorted(unsafe | unknown)}')


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical inputs; reject extras, invalid domains and broken numerics.

    Optional categories may be omitted or missing. Location/project values can be
    unseen. Closed-domain statuses and subtypes must be valid. Coercible numeric
    and date strings are accepted, while malformed values raise clear errors.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Input must be a pandas DataFrame.')
    assert_safe_features(list(frame.columns))
    missing = set(INPUT_COLUMNS) - set(OPTIONAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Input must contain at least one property.')
    x = frame.reindex(columns=INPUT_COLUMNS).copy()
    if x.ACTUAL_AREA.map(lambda v: isinstance(v, (bool, np.bool_))).any():
        raise ValueError('ACTUAL_AREA must be numeric, not boolean.')
    x['ACTUAL_AREA'] = pd.to_numeric(x.ACTUAL_AREA, errors='coerce')
    if not (np.isfinite(x.ACTUAL_AREA) & x.ACTUAL_AREA.gt(0) & x.ACTUAL_AREA.le(MAX_SURFACE)).all():
        raise ValueError(f'ACTUAL_AREA must be finite and in (0, {MAX_SURFACE:g}] square metres.')
    if pd.api.types.is_numeric_dtype(x.INSTANCE_DATE):
        raise ValueError('INSTANCE_DATE must be a date or ISO date string, not a number.')
    try:
        x['INSTANCE_DATE'] = pd.to_datetime(x.INSTANCE_DATE, errors='raise', format='mixed')
        if x.INSTANCE_DATE.dt.tz is not None:
            raise ValueError('Timezone-aware dates are not supported; provide local transaction dates.')
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f'INSTANCE_DATE must contain valid local dates: {exc}') from exc
    if x.INSTANCE_DATE.isna().any() or not x.INSTANCE_DATE.dt.year.between(1900, 2100).all():
        raise ValueError('INSTANCE_DATE must contain valid dates between 1900 and 2100.')
    for column in CATEGORICAL:
        nonmissing = x[column].dropna()
        if not nonmissing.map(lambda v: isinstance(v, str)).all():
            raise ValueError(f'{column} must contain strings or missing values.')
        x[column] = x[column].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
    for column in ['AREA_EN', 'PROP_SB_TYPE_EN']:
        if x[column].eq('Unknown').any():
            raise ValueError(f'{column} is required and cannot be empty.')
    domains = {'PROP_SB_TYPE_EN': RESIDENTIAL_TYPES,
               'IS_OFFPLAN_EN': {'Ready', 'Off-Plan', 'Unknown'},
               'IS_FREE_HOLD_EN': {'Free Hold', 'Non Free Hold', 'Unknown'}}
    for column, allowed in domains.items():
        invalid = set(x[column]) - allowed
        if invalid:
            raise ValueError(f'Invalid {column} values: {sorted(invalid)}')
    def valid_room(value: str) -> bool:
        if value in {'Studio', 'PENTHOUSE', 'Unknown'}:
            return True
        match = re.fullmatch(r'(\d+) B/R', value)
        return bool(match and 0 <= int(match.group(1)) <= 20)
    if not x.ROOMS_EN.map(valid_room).all():
        raise ValueError('ROOMS_EN must be Studio, PENTHOUSE, Unknown, or 0–20 B/R.')
    return x


def validate_training(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate the full training table before selecting the separate target."""
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError('Training data must be a DataFrame with unique column names.')
    if TARGET not in frame:
        raise ValueError(f'Missing target column: {TARGET}')
    x = validate_input(frame.drop(columns=TARGET))
    y = pd.to_numeric(frame[TARGET], errors='coerce').to_numpy(dtype=float)
    if not (np.isfinite(y) & (y > 0)).all():
        raise ValueError(f'{TARGET} must be finite and positive for all formulations.')
    return x, y
