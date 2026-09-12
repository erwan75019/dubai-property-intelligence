"""Streamlit-facing prediction service; all transforms live in the saved estimator."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import logging
import joblib
import numpy as np
import pandas as pd
from src.config import CONFIG, VERSION, INPUT_COLUMNS
from src.evaluation import interval_bounds
from src.schema import validate_input

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class PriceEstimate:
    price_aed: float
    lower_aed: float
    upper_aed: float
    warnings: tuple[str, ...]


def load_artifacts(path: Path = CONFIG.artifact_path) -> tuple[object, dict]:
    """Verify matching metadata before loading a trusted local joblib artifact.

    Hashing detects accidental mismatch, not malicious artifacts. Never load an
    untrusted joblib file: deserialization can execute code.
    """
    path = Path(path)
    metadata_path = path.with_suffix('.metadata.json')
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError('Model or metadata missing. Run python -m src.preprocessing and python -m src.model.')
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata['project_version'] != VERSION:
            raise ValueError('Model version differs from this application; retrain the model.')
        if metadata.get('features') != INPUT_COLUMNS:
            raise ValueError('Model feature schema differs from the application; retrain the model.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['artifact_sha256']:
            raise ValueError('Model and metadata do not match. Regenerate both artifacts together.')
        for key in ['interval', 'categories', 'surface_bounds', 'date_ranges', 'test_metrics']:
            if key not in metadata:
                raise ValueError(f'Missing model metadata: {key}')
        model = joblib.load(path)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f'Unable to read model artifacts: {exc}') from exc
    return model, metadata


def predict_property(model, metadata: dict, values: dict) -> PriceEstimate:
    """Validate a single form record and use the exact persisted prediction path."""
    frame = validate_input(pd.DataFrame([values]))
    prediction = model.predict(frame)
    if np.asarray(prediction).shape != (1,) or not np.isfinite(prediction).all() or prediction[0] <= 0:
        raise ValueError('The model could not produce a valid price for this property.')
    lower, upper = interval_bounds(prediction, metadata['interval'])
    row = frame.iloc[0]
    warnings = []
    if not metadata['surface_bounds'][0] <= row.ACTUAL_AREA <= metadata['surface_bounds'][1]:
        warnings.append('Surface is outside the observed training range; this is extrapolation.')
    start, end = map(pd.Timestamp, metadata['date_ranges']['development'])
    if not start.normalize() <= row.INSTANCE_DATE.normalize() <= end.normalize():
        warnings.append('Transaction date is outside model training dates; market changes may reduce accuracy.')
    for column in ['AREA_EN', 'PROJECT_EN']:
        if row[column] not in metadata['categories'][column]:
            warnings.append(f'{column} was not observed in training; the model uses its unseen-category behavior.')
    if row.PROP_SB_TYPE_EN != 'Flat':
        warnings.append('Villa and rare Residential subtype errors can be much larger than typical apartment errors.')
    if prediction[0] >= 5_000_000:
        warnings.append('High-value property ranges are less reliable: coverage was only about 41% for observed holdout prices above AED 10 million.')
    return PriceEstimate(float(prediction[0]), float(lower[0]), float(upper[0]), tuple(warnings))
