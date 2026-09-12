"""Read-only development-period diagnostic audit; flags never become model inputs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from src.config import CONFIG, RESIDENTIAL_TYPES, SALE_PROCEDURES, INPUT_COLUMNS


def run_audit() -> None:
    """Summarize suspicious scope without choosing target-dependent exclusions."""
    raw = pd.read_csv(CONFIG.raw_path)
    dates = pd.to_datetime(raw.INSTANCE_DATE, errors='coerce')
    data = raw.loc[(dates < pd.Timestamp(CONFIG.calibration_start))
                   & raw.PROP_SB_TYPE_EN.isin(RESIDENTIAL_TYPES)
                   & raw.USAGE_EN.eq('Residential')].copy()
    denominator = data.ACTUAL_AREA.where(data.ACTUAL_AREA.gt(0))
    data['AREA_RATIO'] = (data.PROCEDURE_AREA / denominator).replace([np.inf, -np.inf], np.nan)
    data['PRICE_PER_SQM'] = (data.TRANS_VALUE / denominator).replace([np.inf, -np.inf], np.nan)
    summary = data.groupby('PROCEDURE_EN').agg(rows=('TRANS_VALUE','size'),
        ratio_median=('AREA_RATIO','median'), ratio_mean=('AREA_RATIO','mean'),
        ratio_max=('AREA_RATIO','max'), median_price_per_sqm=('PRICE_PER_SQM','median'))
    summary.to_csv(CONFIG.reports_path / 'v2_procedure_audit.csv')
    counts = {'period': f'Before {CONFIG.calibration_start}; residential subset before cleaning',
        'rows': len(data),
        'development_registration_rows': int(data.PROCEDURE_EN.str.contains('Development', na=False).sum()),
        'ratio_outside_half_to_two': int((data.AREA_RATIO.notna() & ~data.AREA_RATIO.between(.5,2)).sum()),
        'unit_price_below_100_AED': int(data.PRICE_PER_SQM.lt(100).sum()),
        'unit_price_above_100000_AED': int(data.PRICE_PER_SQM.gt(100000).sum()),
        'commercial_room_labels': int(data.ROOMS_EN.isin(['Office','Shop']).sum()),
        'diagnostic_only': '100 and 100000 AED/m² are broad review thresholds, not validity judgments or filters.',
        'input_features': INPUT_COLUMNS, 'sale_procedures': sorted(SALE_PROCEDURES)}
    (CONFIG.reports_path / 'v2_quality_audit.json').write_text(json.dumps(counts, indent=2))
    # Sensitive example records stay in the ignored processed-data directory.
    flag = data.PRICE_PER_SQM.lt(100) | data.PRICE_PER_SQM.gt(100000) | data.AREA_RATIO.gt(2)
    data.loc[flag].to_csv(CONFIG.cleaned_path.parent / 'v2_review_queue.csv', index=False)

if __name__ == '__main__':
    run_audit()
