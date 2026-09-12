"""Central V2 data, feature and experiment contract."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2.0.0'
TARGET = 'TRANS_VALUE'
SEED = 42
RESIDENTIAL_TYPES = {'Flat', 'Villa', 'Residential / Villas', 'Residential Flats', 'Residential / Attached Villas'}
SALE_PROCEDURES = {'Sale', 'Sell - Pre registration', 'Delayed Sell', 'Sale On Payment Plan'}
CATEGORICAL = ['AREA_EN', 'PROP_SB_TYPE_EN', 'ROOMS_EN', 'IS_OFFPLAN_EN', 'IS_FREE_HOLD_EN', 'PROJECT_EN']
INPUT_COLUMNS = ['ACTUAL_AREA', 'INSTANCE_DATE'] + CATEGORICAL
OPTIONAL_COLUMNS = ['ROOMS_EN', 'IS_OFFPLAN_EN', 'IS_FREE_HOLD_EN', 'PROJECT_EN']
NUMERIC = ['ACTUAL_AREA', 'LOG_AREA', 'YEAR', 'MONTH', 'QUARTER']
FORBIDDEN = {'TRANS_VALUE', 'PRICE_PER_SQM', 'AREA_RATIO', 'PROCEDURE_AREA', 'TRANSACTION_NUMBER', 'PROCEDURE_EN', 'predicted_price', 'residual'}
MAX_SURFACE = 1_000_000.0  # Input-unit sanity guard, deliberately far above home sizes.

@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen calendar boundaries shared across all formulations and segments."""
    raw_path: Path = ROOT / 'data/raw/transactions-2026-09-11.csv'
    cleaned_path: Path = ROOT / 'data/processed/residential_sales.csv'
    reports_path: Path = ROOT / 'reports'
    artifact_path: Path = ROOT / 'models/price_pipeline.joblib'
    baseline_path: Path | None = ROOT / 'data/processed/v1_test_predictions.csv'
    validation_start: str = '2026-06-12'
    calibration_start: str = '2026-07-09'
    test_start: str = '2026-07-23'
    seed: int = SEED
    interval_alpha: float = .1

CONFIG = ExperimentConfig()
