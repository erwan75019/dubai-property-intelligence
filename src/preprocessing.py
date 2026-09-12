"""Auditable eligibility rules, independent of learned price thresholds."""
from pathlib import Path
import argparse
import json
import logging
import numpy as np
import pandas as pd
from src.config import CONFIG, INPUT_COLUMNS, ROOT, RESIDENTIAL_TYPES, SALE_PROCEDURES
from src.schema import validate_training

LOGGER = logging.getLogger(__name__)
REQUIRED = {'PROP_SB_TYPE_EN', 'USAGE_EN', 'GROUP_EN', 'PROCEDURE_EN', 'INSTANCE_DATE',
            'TRANS_VALUE', 'ACTUAL_AREA', 'PROCEDURE_AREA', 'AREA_EN'}


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the residential cohort and sequential exclusion counts.

    Preserve V1 sale-procedure and area-ratio assumptions. Reject explicit
    commercial room labels on residential properties. Never trim price quantiles.
    Target-dependent diagnostic flags are reported elsewhere, not used to filter.
    """
    if raw.columns.duplicated().any():
        raise ValueError('Duplicate source column names are not allowed.')
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f'Missing source columns: {sorted(missing)}')
    df = raw.copy()
    audit = {'input_rows': len(df)}
    LOGGER.info('Raw dataset shape: %s', df.shape)
    def keep(mask: pd.Series, reason: str) -> None:
        nonlocal df
        mask = mask.fillna(False)
        audit[reason] = int((~mask).sum())
        LOGGER.info('Filter %s removed %s rows', reason, audit[reason])
        df = df.loc[mask].copy()
    keep(df.PROP_SB_TYPE_EN.isin(RESIDENTIAL_TYPES) & df.USAGE_EN.eq('Residential')
         & df.GROUP_EN.eq('Sales'), 'outside_residential_sales')
    keep(df.PROCEDURE_EN.isin(SALE_PROCEDURES), 'excluded_procedure')
    df['INSTANCE_DATE'] = pd.to_datetime(df.INSTANCE_DATE, errors='coerce')
    for col in ['TRANS_VALUE', 'ACTUAL_AREA', 'PROCEDURE_AREA']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    keep(df.INSTANCE_DATE.notna() & np.isfinite(df.TRANS_VALUE) & df.TRANS_VALUE.gt(0)
         & np.isfinite(df.ACTUAL_AREA) & df.ACTUAL_AREA.gt(0), 'invalid_date_price_or_surface')
    ratio = df.PROCEDURE_AREA / df.ACTUAL_AREA
    keep(df.PROCEDURE_AREA.isna() | (np.isfinite(ratio) & ratio.between(.5, 2)), 'area_scope_mismatch')
    if 'ROOMS_EN' in df:
        keep(~df.ROOMS_EN.isin(['Office', 'Shop']), 'commercial_room_mismatch')
    if 'TRANSACTION_NUMBER' in df:
        # Conflicting ID records cannot safely be resolved by arbitrary selection.
        relevant = [c for c in INPUT_COLUMNS + ['TRANS_VALUE'] if c in df]
        conflicts = df.groupby('TRANSACTION_NUMBER')[relevant].nunique(dropna=False).max(axis=1)
        keep(~df.TRANSACTION_NUMBER.isin(conflicts[conflicts.gt(1)].index), 'conflicting_transaction_id')
        keep(~(df.TRANSACTION_NUMBER.notna() & df.TRANSACTION_NUMBER.duplicated()), 'duplicate_transaction_id')
    df = df.reindex(columns=INPUT_COLUMNS + ['TRANS_VALUE']).sort_values('INSTANCE_DATE').reset_index(drop=True)
    # Fail loudly for remaining schema/domain problems rather than silently dropping.
    validate_training(df)
    audit['output_rows'] = len(df)
    LOGGER.info('Clean dataset shape: %s', df.shape)
    return df, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=CONFIG.raw_path)
    parser.add_argument('--output', type=Path, default=CONFIG.cleaned_path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    df, audit = clean_transactions(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    args.output.with_name('cleaning_audit.json').write_text(json.dumps(audit, indent=2))
    LOGGER.info('Clean dataset written: %s', args.output)

if __name__ == '__main__':
    main()
