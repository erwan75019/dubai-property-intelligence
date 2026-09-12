"""Behavioral ML engineering tests; no private dataset or full training required."""
from dataclasses import replace
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.config import CONFIG, VERSION, INPUT_COLUMNS, NUMERIC, CATEGORICAL, FORBIDDEN
from src.features import FeatureBuilder
from src.schema import validate_input, validate_training, assert_safe_features
from src.model import fit_model
from src.benchmark import make_model
from src.estimators import SegmentedPriceModel
from src.evaluation import (temporal_split, chronological_partitions, date_cv, metrics,
                            calibrate_interval, interval_bounds)
from src.preprocessing import clean_transactions
from src.inference import predict_property, load_artifacts

@pytest.fixture
def raw():
    return pd.DataFrame([dict(TRANSACTION_NUMBER=str(i), PROP_SB_TYPE_EN='Flat', USAGE_EN='Residential',
        GROUP_EN='Sales', PROCEDURE_EN='Sale', INSTANCE_DATE=f'2026-01-{i+1:02d}',
        TRANS_VALUE=1_000_000+i*10000, ACTUAL_AREA=100.+i, PROCEDURE_AREA=100.+i,
        AREA_EN='Example', ROOMS_EN='1 B/R', IS_OFFPLAN_EN='Ready',
        IS_FREE_HOLD_EN='Free Hold', PROJECT_EN=None) for i in range(20)])

@pytest.fixture
def clean(raw):
    return clean_transactions(raw)[0]


def test_cleaning_preserves_luxury_and_excludes_mismatch(raw):
    raw.loc[0, 'TRANS_VALUE'] = 100_000_000
    raw.loc[1, 'PROCEDURE_EN'] = 'Development Registration'
    raw.loc[2, 'PROCEDURE_AREA'] = 10_000
    raw.loc[3, 'ACTUAL_AREA'] = 0
    raw.loc[4, 'INSTANCE_DATE'] = 'bad'
    clean, audit = clean_transactions(raw)
    assert len(clean) == 16
    assert clean.TRANS_VALUE.max() == 100_000_000
    assert audit['area_scope_mismatch'] == 1
    assert sum(v for k,v in audit.items() if k not in {'input_rows','output_rows'}) + len(clean) == len(raw)


def test_cleaning_schema_and_nan_categories(clean):
    assert list(clean.columns) == INPUT_COLUMNS + ['TRANS_VALUE']
    assert pd.api.types.is_datetime64_any_dtype(clean.INSTANCE_DATE)
    assert FeatureBuilder().fit_transform(clean[INPUT_COLUMNS]).PROJECT_EN.eq('Unknown').all()


def test_duplicates(raw):
    clean, audit = clean_transactions(pd.concat([raw, raw.iloc[[0]]]))
    assert len(clean) == 20 and audit['duplicate_transaction_id'] == 1

@pytest.mark.parametrize('column,value', [('TRANS_VALUE', 9), ('ACTUAL_AREA', 101)])
def test_conflicting_transaction(raw, column, value):
    duplicate = raw.iloc[[0]].copy()
    duplicate[column] = value
    clean, audit = clean_transactions(pd.concat([raw, duplicate]))
    assert len(clean) == 19 and audit['conflicting_transaction_id'] == 2


def test_commercial_room_mismatch(raw):
    raw.loc[0, 'ROOMS_EN'] = 'Office'
    clean, audit = clean_transactions(raw)
    assert len(clean) == 19 and audit['commercial_room_mismatch'] == 1


def test_missing_source(raw):
    with pytest.raises(ValueError, match='Missing source'):
        clean_transactions(raw.drop(columns='PROCEDURE_EN'))

@pytest.mark.parametrize('value', [0, -1, np.inf, -np.inf, None, 'invalid', True, 1_000_001])
def test_invalid_surface(clean, value):
    x = clean[INPUT_COLUMNS].copy()
    x['ACTUAL_AREA'] = value
    with pytest.raises(ValueError, match='ACTUAL_AREA'):
        validate_input(x)

@pytest.mark.parametrize('column', ['ACTUAL_AREA', 'INSTANCE_DATE', 'AREA_EN', 'PROP_SB_TYPE_EN'])
def test_missing_schema(clean, column):
    with pytest.raises(ValueError, match='Missing input'):
        validate_input(clean[INPUT_COLUMNS].drop(columns=column))

@pytest.mark.parametrize('column,value', [('IS_OFFPLAN_EN','Finished'), ('IS_FREE_HOLD_EN','Yes'),
    ('PROP_SB_TYPE_EN','Office'), ('ROOMS_EN','Shop'), ('ROOMS_EN','99 B/R'), ('AREA_EN',1)])
def test_invalid_categories(clean, column, value):
    x = clean[INPUT_COLUMNS].copy()
    x[column] = value
    with pytest.raises(ValueError):
        validate_input(x)

@pytest.mark.parametrize('column', ['AREA_EN', 'PROP_SB_TYPE_EN'])
def test_missing_required_category(clean, column):
    x = clean[INPUT_COLUMNS].copy()
    x[column] = None
    with pytest.raises(ValueError, match='cannot be empty'):
        validate_input(x)

@pytest.mark.parametrize('value', ['not a date', None, 20260101, '2200-01-01', '2026-01-01T00:00:00+00:00'])
def test_invalid_dates(clean, value):
    x = clean[INPUT_COLUMNS].copy()
    x['INSTANCE_DATE'] = value
    with pytest.raises(ValueError, match='INSTANCE_DATE'):
        validate_input(x)

@pytest.mark.parametrize('column', sorted(FORBIDDEN) + ['mysterious_target_derived_column'])
def test_leakage_rejected(clean, column):
    x = clean[INPUT_COLUMNS].assign(**{column: 99})
    with pytest.raises(ValueError, match='Forbidden or unexpected'):
        FeatureBuilder().fit_transform(x)
    assert not set(NUMERIC+CATEGORICAL) & FORBIDDEN


def test_duplicate_features_and_target_missing(clean):
    with pytest.raises(ValueError, match='Duplicate feature'):
        validate_input(pd.concat([clean[INPUT_COLUMNS], clean[['ACTUAL_AREA']]], axis=1))
    with pytest.raises(ValueError, match='Missing target'):
        validate_training(clean.drop(columns='TRANS_VALUE'))
    with pytest.raises(ValueError, match='unique column'):
        validate_training(pd.concat([clean, clean[['TRANS_VALUE']]], axis=1))

@pytest.mark.parametrize('value', [0, -2, np.nan, np.inf])
def test_invalid_training_target(clean, value):
    clean['TRANS_VALUE'] = value
    with pytest.raises(ValueError, match='finite and positive'):
        validate_training(clean)


def test_missing_optional_and_unseen_categories(clean):
    model = fit_model('ridge', clean)
    x = clean[INPUT_COLUMNS].drop(columns=['PROJECT_EN', 'ROOMS_EN', 'IS_OFFPLAN_EN', 'IS_FREE_HOLD_EN'])
    x['AREA_EN'] = 'Never observed location'
    prediction = model.predict(x)
    assert prediction.shape == (len(clean),) and np.isfinite(prediction).all()
    x['PROJECT_EN'] = 'Never observed project'
    assert np.isfinite(model.predict(x)).all()


def test_feature_order_and_roundtrip(clean, tmp_path):
    model = fit_model('ridge', clean)
    x = clean[INPUT_COLUMNS]
    expected = model.predict(x)
    np.testing.assert_allclose(expected, model.predict(x[x.columns[::-1]]))
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(expected, joblib.load(path).predict(x))


def test_training_vocabulary_is_not_changed_by_inference(clean):
    model = fit_model('ridge', clean)
    encoder = model.pipeline_.named_steps['preprocess'].named_steps['encode'].named_transformers_['area']
    before = encoder.categories_[0].copy()
    model.predict(clean[INPUT_COLUMNS].assign(AREA_EN='Holdout-only area'))
    np.testing.assert_array_equal(before, encoder.categories_[0])
    assert 'Holdout-only area' not in before

@pytest.mark.parametrize('form', ['direct','log','per_sqm'])
def test_formulation_reconstruction_and_reproducibility(clean, form):
    model = fit_model('dummy', clean, form)
    x = clean[INPUT_COLUMNS]
    expected = np.median(clean.TRANS_VALUE)
    if form == 'log':
        expected = np.expm1(np.median(np.log1p(clean.TRANS_VALUE)))
    elif form == 'per_sqm':
        expected = np.median(clean.TRANS_VALUE / clean.ACTUAL_AREA) * clean.ACTUAL_AREA.to_numpy()
    np.testing.assert_allclose(model.predict(x), expected)
    np.testing.assert_allclose(model.predict(x), fit_model('dummy', clean, form).predict(x))


def test_native_catboost_clone_unseen_and_repeatable(clean):
    model = make_model('catboost', 'log', {'iterations': 5, 'depth': 2})
    a = clone(model).fit(clean[INPUT_COLUMNS], clean.TRANS_VALUE)
    b = clone(model).fit(clean[INPUT_COLUMNS], clean.TRANS_VALUE)
    x = clean[INPUT_COLUMNS].assign(PROJECT_EN='New project', AREA_EN='New area')
    np.testing.assert_allclose(a.predict(x), b.predict(x))


def test_temporal_split_keeps_whole_dates(clean):
    data = pd.concat([clean, clean.assign(INSTANCE_DATE=clean.INSTANCE_DATE + pd.Timedelta(hours=4))])
    train, test = temporal_split(data)
    assert train.INSTANCE_DATE.max() < test.INSTANCE_DATE.min()
    assert set(train.INSTANCE_DATE.dt.date).isdisjoint(test.INSTANCE_DATE.dt.date)
    assert len(train) + len(test) == len(data)


def test_four_periods_and_cv(clean):
    data = pd.concat([clean]*13, ignore_index=True)
    data['INSTANCE_DATE'] = pd.date_range('2026-01-01', periods=len(data), freq='D')
    parts = chronological_partitions(data)
    assert sum(map(len, parts.values())) == len(data)
    for left, right in zip(list(parts.values())[:-1], list(parts.values())[1:]):
        assert left.INSTANCE_DATE.max() < right.INSTANCE_DATE.min()
    for tr, va in date_cv(parts['train']):
        assert parts['train'].iloc[tr].INSTANCE_DATE.max() < parts['train'].iloc[va].INSTANCE_DATE.min()
    with pytest.raises(ValueError, match='strictly increasing'):
        chronological_partitions(data, replace(CONFIG, test_start='2026-01-01'))


def test_interval_calibration_and_bounds():
    y = np.linspace(1e6, 2e6, 50)
    prediction = y*.9
    calibration = calibrate_interval(y, prediction)
    lower, upper = interval_bounds(prediction, calibration)
    assert np.all(lower < prediction) and np.all(upper > prediction)
    assert np.isclose(calibration['log_radius'], abs(np.log(.9)))
    with pytest.raises(ValueError, match='too small'):
        calibrate_interval([100.], [100.])


def test_metrics_invalid_and_singleton():
    with pytest.raises(ValueError, match='finite'):
        metrics([10], [np.inf])
    assert metrics([10], [11])['R2'] is None


def test_routing_shape_and_fallback(clean):
    global_model = fit_model('dummy', clean)
    specialist = fit_model('ridge', clean)
    routed = SegmentedPriceModel(global_model, {'Flat': specialist})
    np.testing.assert_allclose(routed.predict(clean[INPUT_COLUMNS]), specialist.predict(clean[INPUT_COLUMNS]))
    villa = clean[INPUT_COLUMNS].assign(PROP_SB_TYPE_EN='Villa')
    np.testing.assert_allclose(routed.predict(villa), global_model.predict(villa))


def test_streamlit_service_and_artifact_integrity(clean, tmp_path):
    model = fit_model('ridge', clean)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    metadata = {'project_version': VERSION, 'features': INPUT_COLUMNS, 'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'interval': {'log_radius': .2}, 'categories': {'AREA_EN':['Example'], 'PROJECT_EN':[]},
        'surface_bounds':[100,119], 'date_ranges': {'development':['2026-01-01','2026-01-20']}, 'test_metrics':{}}
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    loaded, meta = load_artifacts(path)
    values = clean[INPUT_COLUMNS].iloc[0].to_dict()
    estimate = predict_property(loaded, meta, values)
    assert estimate.lower_aed < estimate.price_aed < estimate.upper_aed
    assert np.isclose(estimate.price_aed, model.predict(clean[INPUT_COLUMNS].iloc[[0]])[0])
    with pytest.raises(ValueError, match='ACTUAL_AREA'):
        predict_property(loaded, meta, {**values, 'ACTUAL_AREA': -1})
    metadata['artifact_sha256'] = 'wrong'
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='do not match'):
        load_artifacts(path)


def test_end_to_end_orchestration_without_expensive_search(clean, tmp_path, monkeypatch):
    """Exercise real refits, final reports, intervals and service with cheap trials."""
    import src.model as training
    data = pd.concat([clean]*13, ignore_index=True)
    data['INSTANCE_DATE'] = pd.date_range('2026-01-01', periods=len(data), freq='D')
    def cheap_selection(train, validation, out):
        assert train.INSTANCE_DATE.max() < validation.INSTANCE_DATE.min()
        assert validation.INSTANCE_DATE.max() < pd.Timestamp(CONFIG.calibration_start)
        records = [dict(candidate=f'dummy__{form}', model='dummy', formulation=form,
                        parameters={}, MAE=1., RMSE=2., R2=0., fit_seconds=0.)
                   for form in ['direct','log','per_sqm']]
        winner = records[0]
        selection = {'global': winner, 'use_routing': False, 'holdout_used_for_selection': False}
        return records, winner, {}, {}, False, selection
    monkeypatch.setattr(training, 'select_candidates', cheap_selection)
    config = replace(CONFIG, cleaned_path=tmp_path/'clean.csv', reports_path=tmp_path/'reports',
                     artifact_path=tmp_path/'model.joblib', baseline_path=None)
    data.to_csv(config.cleaned_path, index=False)
    model, scores, report = training.run_training(data, config)
    assert report['selection']['holdout_used_for_selection'] is False
    assert pd.Timestamp(report['date_ranges']['development'][1]) < pd.Timestamp(CONFIG.calibration_start)
    assert report['interval']['n'] == report['rows']['calibration']
    loaded, metadata = load_artifacts(config.artifact_path)
    row = data[INPUT_COLUMNS].iloc[[0]]
    np.testing.assert_allclose(model.predict(row), loaded.predict(row))
    assert predict_property(loaded, metadata, row.iloc[0].to_dict()).price_aed > 0
    assert (config.reports_path/'v2_comparison.csv').exists()


def test_selection_ties_are_deterministic_and_not_runtime_based():
    from src.model import select_record
    a = dict(candidate='a', MAE=10., RMSE=20., fit_seconds=999.)
    b = dict(candidate='b', MAE=10., RMSE=20., fit_seconds=.1)
    assert select_record([b,a])['candidate'] == 'a'


def test_training_entry_point_rejects_leakage(clean):
    with pytest.raises(ValueError, match='Forbidden or unexpected'):
        fit_model('ridge', clean.assign(PRICE_PER_SQM=1000))


def test_independent_splits_do_not_change_with_test_targets(clean):
    data = pd.concat([clean]*13, ignore_index=True)
    data['INSTANCE_DATE'] = pd.date_range('2026-01-01',periods=len(data),freq='D')
    first = chronological_partitions(data)
    changed = data.copy()
    changed.loc[changed.INSTANCE_DATE.ge(CONFIG.test_start),'TRANS_VALUE'] *= 999
    second = chronological_partitions(changed)
    for name in ['train','validation','calibration']:
        pd.testing.assert_frame_equal(first[name],second[name])
