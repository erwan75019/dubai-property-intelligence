"""V2 CLI: frozen temporal selection, calibration, final audit and persistence."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import logging
import platform
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from src.config import CONFIG, VERSION, INPUT_COLUMNS, CATEGORICAL, ExperimentConfig
from src.schema import validate_training
from src.benchmark import benchmark, tune, make_model, fit_model, FORMULATIONS, model_families
from src.estimators import SegmentedPriceModel
from src.evaluation import (metrics, temporal_split, chronological_partitions, calibrate_interval,
                            interval_bounds, error_tables, paired_day_bootstrap)

LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def select_record(records: list[dict]) -> dict:
    """Predeclared ranking: validation MAE, then RMSE; never holdout ranking."""
    return min(records, key=lambda r: (r['MAE'], r['RMSE'], r['candidate']))


def compare_specialists(train, validation, records, fitted) -> tuple[dict, dict, list[dict]]:
    """Compare specialists against the global model on exactly the same rows."""
    global_record = select_record(records)
    global_model = fitted[global_record['candidate']]
    specifications = [select_record([r for r in records if r['formulation'] == form]) for form in FORMULATIONS]
    selected, selected_models, report = {}, {}, []
    for segment in ['Flat', 'Villa']:
        tr = train.loc[train.PROP_SB_TYPE_EN.eq(segment)]
        va = validation.loc[validation.PROP_SB_TYPE_EN.eq(segment)]
        if len(tr) < 100 or len(va) < 30:
            LOGGER.info('Insufficient observations for %s specialist; using global fallback', segment)
            continue
        global_scores = metrics(va.TRANS_VALUE, global_model.predict(va[INPUT_COLUMNS]))
        local_records, local_models = [], {}
        for spec in specifications:
            LOGGER.info('Segment %s: %s', segment, spec['candidate'])
            model = make_model(spec['model'], spec['formulation'], spec['parameters'])
            start = time.monotonic()
            model.fit(tr[INPUT_COLUMNS], tr.TRANS_VALUE)
            result = {**spec, 'fit_seconds': time.monotonic()-start,
                      **metrics(va.TRANS_VALUE, model.predict(va[INPUT_COLUMNS]))}
            local_records.append(result)
            local_models[spec['candidate']] = model
            report.append({'segment': segment, 'candidate': spec['candidate'], 'scope': 'specialist',
                           **metrics(va.TRANS_VALUE, model.predict(va[INPUT_COLUMNS]))})
        best = select_record(local_records)
        # Practicality gate: require 2% less MAE, with at most 10% RMSE deterioration.
        use = best['MAE'] < .98 * global_scores['MAE'] and best['RMSE'] <= 1.1 * global_scores['RMSE']
        selected[segment] = {**best, 'use_specialist': use}
        selected_models[segment] = local_models[best['candidate']]
        report.append({'segment': segment, 'candidate': global_record['candidate'], 'scope': 'global', **global_scores})
    return selected, selected_models, report


def refit_record(record: dict, train: pd.DataFrame):
    return make_model(record['model'], record['formulation'], record['parameters']).fit(train[INPUT_COLUMNS], train.TRANS_VALUE)


def select_candidates(train: pd.DataFrame, validation: pd.DataFrame, out: Path) -> tuple:
    """Benchmark, tune, ablate and freeze routing using development labels only."""
    records, fitted = benchmark(train, validation)
    write_json(out / 'v2_benchmark_validation.json', records)
    # Tune only two best nontrivial families, using early training folds.
    family_best = [select_record([r for r in records if r['model'] == family])
                   for family in model_families() if family != 'dummy']
    for base in sorted(family_best, key=lambda r: r['MAE'])[:2]:
        record, model, cv = tune(base['model'], base['formulation'], train, validation)
        records.append(record)
        fitted[record['candidate']] = model
        cv.to_csv(out / f"v2_cv_{base['model']}.csv", index=False)
    write_json(out / 'v2_validation.json', records)
    pd.DataFrame(records).to_csv(out / 'validation_metrics.csv', index=False)
    winner = select_record(records)
    LOGGER.info('Global selection (validation only): %s', winner['candidate'])

    # Feature ablations are diagnostics only: the full feature specification is frozen.
    ablations = []
    for label, log_area, project in [('no_log_area', False, True), ('no_project', True, False)]:
        model = make_model(winner['model'], winner['formulation'], winner['parameters'], log_area, project)
        model.fit(train[INPUT_COLUMNS], train.TRANS_VALUE)
        ablations.append({'features': label, **metrics(validation.TRANS_VALUE, model.predict(validation[INPUT_COLUMNS]))})
    ablations.append({'features': 'full', **{k: winner[k] for k in ['MAE', 'RMSE', 'R2']}})
    pd.DataFrame(ablations).to_csv(out / 'v2_feature_ablations.csv', index=False)

    specialist_specs, specialist_models, segment_validation = compare_specialists(train, validation, records, fitted)
    pd.DataFrame(segment_validation).to_csv(out / 'v2_segment_validation.csv', index=False)
    active = {s: specialist_models[s] for s, spec in specialist_specs.items() if spec['use_specialist']}
    routed = SegmentedPriceModel(fitted[winner['candidate']], active)
    routed_metrics = metrics(validation.TRANS_VALUE, routed.predict(validation[INPUT_COLUMNS]))
    use_routing = bool(active) and routed_metrics['MAE'] < .98 * winner['MAE'] and routed_metrics['RMSE'] <= 1.1 * winner['RMSE']
    selection = {'global': winner, 'specialists': specialist_specs, 'use_routing': use_routing,
                 'routing_validation_metrics': routed_metrics,
                 'policy': 'Validation MAE primary, RMSE tie-break; specialists need >=2% MAE improvement with <=10% RMSE deterioration; routed model must pass the same overall gate.',
                 'holdout_used_for_selection': False}
    write_json(out / 'v2_selection.json', selection)  # Freeze before any final-test prediction.

    return records, winner, specialist_specs, active, use_routing, selection


def refit_selected(records: list[dict], winner: dict, specialist_specs: dict,
                   active: dict, use_routing: bool, development: pd.DataFrame) -> tuple:
    """Fit frozen candidates through July 8, leaving calibration untouched."""
    # Refit just the winning candidate for each formulation, not every trial.
    final_globals, formulation_specs = {}, {}
    for form in FORMULATIONS:
        spec = select_record([r for r in records if r['formulation'] == form])
        formulation_specs[form] = spec
        LOGGER.info('Refitting global %s through %s', spec['candidate'], development.INSTANCE_DATE.max())
        final_globals[form] = refit_record(spec, development)
    global_final = final_globals[winner['formulation']]
    final_specialists = {s: refit_record(spec, development.loc[development.PROP_SB_TYPE_EN.eq(s)])
                         for s, spec in specialist_specs.items()}
    deployed = SegmentedPriceModel(global_final, {s: final_specialists[s] for s in active}) if use_routing else global_final
    return deployed, final_globals, global_final, final_specialists, formulation_specs


def write_comparisons(test: pd.DataFrame, prediction: np.ndarray, lower: np.ndarray,
                      upper: np.ndarray, final_globals: dict, global_final,
                      final_specialists: dict, formulation_specs: dict,
                      specialist_specs: dict, use_routing: bool, out: Path) -> None:
    """Final holdout diagnostics cannot alter the frozen selection."""
    comparison = []
    for form, model in final_globals.items():
        comparison.append({'model': formulation_specs[form]['model'], 'formulation': 'global', 'target': form,
                           'split_strategy': 'chronological future holdout', 'rows': len(test),
                           **metrics(test.TRANS_VALUE, model.predict(test[INPUT_COLUMNS]))})
    segment_comparison = []
    for segment, local in final_specialists.items():
        group = test.loc[test.PROP_SB_TYPE_EN.eq(segment)]
        for scope, model in [('global', global_final), ('specialist', local)]:
            score = metrics(group.TRANS_VALUE, model.predict(group[INPUT_COLUMNS]))
            segment_comparison.append({'segment': segment, 'scope': scope, 'rows': len(group), **score})
        comparison.append({'model': specialist_specs[segment]['model'], 'formulation': f'{segment}-only',
                           'target': specialist_specs[segment]['formulation'], 'split_strategy': 'same future dates, segment subset',
                           'rows': len(group), **metrics(group.TRANS_VALUE, local.predict(group[INPUT_COLUMNS]))})
    if use_routing:
        targets = {global_final.formulation} | {s['formulation'] for s in specialist_specs.values() if s['use_specialist']}
        routed_target = next(iter(targets)) if len(targets) == 1 else 'mixed'
        comparison.append({'model': 'segmented ensemble', 'formulation': 'routed', 'target': routed_target,
                           'split_strategy': 'chronological future holdout', 'rows': len(test), **metrics(test.TRANS_VALUE, prediction)})
    pd.DataFrame(segment_comparison).to_csv(out / 'v2_segment_holdout.csv', index=False)
    pd.DataFrame(comparison).to_csv(out / 'v2_comparison.csv', index=False)
    for name, table in error_tables(test, prediction, lower, upper).items():
        table.to_csv(out / f'v2_errors_{name}.csv', index=False)
        if name == 'PROP_SB_TYPE_EN':
            table.rename(columns={'group': 'property_type'}).to_csv(out / 'test_by_property_type.csv', index=False)


def run_training(df: pd.DataFrame, config: ExperimentConfig = CONFIG):
    """Select before reading holdout targets; no test-dependent retraining."""
    validate_training(df)
    parts = chronological_partitions(df, config)
    train, validation, calibration, test = [parts[k] for k in ['train', 'validation', 'calibration', 'test']]
    out = config.reports_path
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.info('Partition sizes: %s', {key: len(value) for key, value in parts.items()})
    records, winner, specialist_specs, active, use_routing, selection = select_candidates(train, validation, out)

    development = pd.concat([train, validation], ignore_index=True).sort_values('INSTANCE_DATE')
    deployed, final_globals, global_final, final_specialists, formulation_specs = refit_selected(
        records, winner, specialist_specs, active, use_routing, development)

    interval = calibrate_interval(calibration.TRANS_VALUE, deployed.predict(calibration[INPUT_COLUMNS]), config.interval_alpha)

    prediction = deployed.predict(test[INPUT_COLUMNS])
    lower, upper = interval_bounds(prediction, interval)
    write_comparisons(test, prediction, lower, upper, final_globals, global_final,
                      final_specialists, formulation_specs, specialist_specs, use_routing, out)

    report = {'project_version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'selected_model': 'segmented ensemble' if use_routing else winner['model'],
              'target': winner['formulation'], 'prediction_unit': 'total price AED', 'selection': selection,
              'output_target': 'TRANS_VALUE', 'prediction_floor_AED': 1.0,
              'model_configuration': {
                  'global': global_final.pipeline_.named_steps['regressor'].get_params(deep=False),
                  'specialists': {s: m.pipeline_.named_steps['regressor'].get_params(deep=False)
                                  for s, m in final_specialists.items()}},
              'processed_input_sha256': hashlib.sha256(config.cleaned_path.read_bytes()).hexdigest() if config.cleaned_path.exists() else None,
              'features': INPUT_COLUMNS, 'engineered_features': ['LOG_AREA', 'YEAR', 'MONTH', 'QUARTER'],
              'test_metrics': metrics(test.TRANS_VALUE, prediction), 'interval': interval,
              'interval_test_coverage': float(np.mean((test.TRANS_VALUE >= lower) & (test.TRANS_VALUE <= upper))),
              'median_interval_width_AED': float(np.median(upper-lower)),
              'rows': {k: len(p) for k, p in parts.items()},
              'date_ranges': {k: [str(p.INSTANCE_DATE.min()), str(p.INSTANCE_DATE.max())] for k, p in {**parts, 'development': development}.items()},
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
              'configuration': asdict(config),
              'external_validation_caveat': 'The V1 holdout was previously inspected. V2 selection is frozen without its labels; a newer export is needed for an untouched external test.'}
    import catboost
    report['versions']['catboost'] = catboost.__version__
    baseline_path = config.baseline_path
    if baseline_path is not None and baseline_path.exists():
        old = pd.read_csv(baseline_path, parse_dates=['INSTANCE_DATE'])
        # The existing holdout is unchanged. Never assume positional alignment.
        key_cols = INPUT_COLUMNS + ['TRANS_VALUE']
        left = test[key_cols].copy()
        right = old.copy()
        left['_occurrence'] = left.groupby(key_cols, dropna=False).cumcount()
        right['_occurrence'] = right.groupby(key_cols, dropna=False).cumcount()
        aligned = left.merge(right[key_cols + ['_occurrence', 'predicted_price']], on=key_cols + ['_occurrence'], how='left', validate='one_to_one', sort=False)
        if len(aligned) == len(test) and aligned.predicted_price.notna().all():
            old_scores = metrics(test.TRANS_VALUE, aligned.predicted_price)
            report['v1_same_rows'] = old_scores
            report['improvement_pct'] = {m: 100*(old_scores[m]-report['test_metrics'][m])/old_scores[m] for m in ['MAE', 'RMSE']}
            report['paired_comparison'] = paired_day_bootstrap(test, aligned.predicted_price, prediction)
            comparisons = pd.read_csv(out / 'v2_comparison.csv')
            v1 = {'model': 'V1 hist_gradient_boosting', 'formulation': 'global', 'target': 'direct',
                  'split_strategy': 'same chronological future holdout', 'rows': len(test), **old_scores}
            pd.concat([pd.DataFrame([v1]), comparisons], ignore_index=True).to_csv(out / 'v2_comparison.csv', index=False)
        else:
            LOGGER.warning('V1 predictions do not align with all current holdout rows; omitting paired comparison.')
    config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed, config.artifact_path, compress=3)
    metadata = {**report, 'artifact_sha256': hashlib.sha256(config.artifact_path.read_bytes()).hexdigest(),
                'categories': {c: sorted(development[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL},
                'surface_bounds': [float(development.ACTUAL_AREA.min()), float(development.ACTUAL_AREA.max())]}
    write_json(config.artifact_path.with_suffix('.metadata.json'), metadata)
    write_json(out / 'evaluation.json', report)
    # Row-level predictions remain excluded from version control.
    test.assign(predicted_price=prediction, lower=lower, upper=upper).to_csv(config.cleaned_path.parent / 'test_predictions.csv', index=False)
    LOGGER.info('Final metrics: %s', report['test_metrics'])
    LOGGER.info('Saved model %s and metadata %s', config.artifact_path, config.artifact_path.with_suffix('.metadata.json'))
    return deployed, pd.DataFrame(records), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=CONFIG.cleaned_path)
    parser.add_argument('--artifact', type=Path, default=CONFIG.artifact_path)
    parser.add_argument('--reports', type=Path, default=CONFIG.reports_path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = replace(CONFIG, cleaned_path=args.input, artifact_path=args.artifact, reports_path=args.reports)
    df = pd.read_csv(args.input, parse_dates=['INSTANCE_DATE'])
    LOGGER.info('Training input shape: %s', df.shape)
    run_training(df, config)

if __name__ == '__main__':
    main()
