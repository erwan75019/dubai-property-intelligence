"""Compact portfolio UI around the unchanged persisted V2 prediction service."""
from pathlib import Path
import logging
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import CONFIG, MAX_SURFACE
from src.inference import load_artifacts, predict_property

LOGGER = logging.getLogger(__name__)
# Presentation defaults from a real development-period row; never used to fit a model.
DEMO = dict(AREA_EN='JUMEIRAH VILLAGE CIRCLE', PROP_SB_TYPE_EN='Flat',
            ROOMS_EN='2 B/R', IS_OFFPLAN_EN='Off-Plan', IS_FREE_HOLD_EN='Free Hold',
            PROJECT_EN='Stax', ACTUAL_AREA=101.16, INSTANCE_DATE='2026-06-05')

st.set_page_config(page_title='Dubai Property Intelligence', page_icon='🏙️',
                   layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
    .block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 1rem; }
</style>''', unsafe_allow_html=True)
st.title('Dubai Property Intelligence')
st.caption('Residential sale estimates · Official Dubai Land Department data · CatBoost V2')


@st.cache_resource
def cached_artifacts(artifact_modified: float, metadata_modified: float):
    """File timestamps invalidate cached models after a retrain."""
    return load_artifacts()


try:
    model, metadata = cached_artifacts(CONFIG.artifact_path.stat().st_mtime,
                                       CONFIG.artifact_path.with_suffix('.metadata.json').stat().st_mtime)
except Exception:
    LOGGER.exception('Could not load model artifacts')
    st.error('The model is unavailable or its files are inconsistent. Run python -m src.preprocessing, then python -m src.model.')
    st.stop()

choices = metadata['categories']


def category_input(label: str, column: str, optional: bool = False) -> str:
    """Select a known demo value, with a safe UI fallback after artifact changes."""
    options = list(dict.fromkeys(choices[column] + (['Unknown'] if optional else [])))
    default = DEMO[column]
    return st.selectbox(label, options, index=options.index(default) if default in options else 0)


inputs_column, result_column = st.columns([1.2, 1], gap='large')
with inputs_column:
    st.subheader('Example property')
    st.caption('Pre-filled demo input from the training data. Edit the details and select Estimate price.')
    with st.form('property'):
        left, right = st.columns(2)
        with left:
            area = category_input('Area', 'AREA_EN')
            subtype = category_input('Property type', 'PROP_SB_TYPE_EN')
            status = category_input('Completion status', 'IS_OFFPLAN_EN', optional=True)
            surface = st.number_input('Surface (m²)', min_value=1.0, max_value=MAX_SURFACE,
                                      value=DEMO['ACTUAL_AREA'], step=1.0, format='%.2f')
        with right:
            project = category_input('Project', 'PROJECT_EN', optional=True)
            rooms = category_input('Bedrooms / room category', 'ROOMS_EN', optional=True)
            tenure = category_input('Ownership', 'IS_FREE_HOLD_EN', optional=True)
            date = st.date_input('Transaction date', value=pd.Timestamp(DEMO['INSTANCE_DATE']).date())
        with st.expander('Enter an unlisted area or project', expanded=False):
            new_area = st.text_input('Unlisted area (optional)',
                                     help='Overrides the selected area. Use the official name; unseen areas are less reliable.')
            new_project = st.text_input('Unlisted project (optional)',
                                        help='Overrides the selected project. Leave blank to use the selection above.')
        st.form_submit_button('Estimate price', type='primary', width='stretch')

# Forms submit their values together. On first load, this same service displays
# the demo estimate automatically; no prices or intervals are hardcoded in the UI.
values = dict(AREA_EN=new_area.strip() or area, PROP_SB_TYPE_EN=subtype, ROOMS_EN=rooms,
              IS_OFFPLAN_EN=status, IS_FREE_HOLD_EN=tenure, PROJECT_EN=new_project.strip() or project,
              ACTUAL_AREA=surface, INSTANCE_DATE=str(date))
with result_column:
    st.subheader('Price estimate')
    st.caption('Demo input · known training example' if values == DEMO else 'Estimate for your submitted property')
    try:
        estimate = predict_property(model, metadata, values)
        with st.container(border=True):
            st.metric('Estimated sale price', f'AED {estimate.price_aed:,.0f}')
            st.markdown('**Empirical prediction range**')
            st.markdown(f'### AED {estimate.lower_aed:,.0f} – {estimate.upper_aed:,.0f}')
            st.caption('An estimated recorded sale amount—not a guaranteed market value.')
        st.caption(f"The range targets {1-metadata['interval']['alpha']:.0%} coverage; measured holdout coverage was {metadata['interval_test_coverage']:.1%}. It is an empirical range, not a guaranteed confidence interval.")
        st.caption('Edit the form and select Estimate price to refresh this result.')
        for warning in estimate.warnings:
            # Unknown is the missing-project option, not an entered unseen name.
            # Keep all other service warnings, including genuine unseen names.
            if values['PROJECT_EN'] == 'Unknown' and warning.startswith('PROJECT_EN was not observed'):
                continue
            st.warning(warning)
    except ValueError as exc:
        st.error(f'Please check the property details: {exc}')
    except Exception:
        LOGGER.exception('Prediction failed')
        st.error('The estimate could not be generated. Please check your inputs or retrain the model if the problem persists.')

with st.expander('Model performance and limitations', expanded=False):
    st.write(f"Model: {metadata['selected_model']}. Future-period MAE: AED {metadata['test_metrics']['MAE']:,.0f}; RMSE: AED {metadata['test_metrics']['RMSE']:,.0f}.")
    st.write('The empirical range is calibrated on a separate earlier period. Coverage varies by area, property type and price; luxury-property ranges are particularly unreliable.')
    st.write('Models were selected on earlier validation transactions, before final comparison. The V1 holdout had already been inspected; a newer dataset is still needed for independent external validation. Floor, view, condition, ownership share and exact unit identifiers are unavailable. Scope mismatches may remain in broad Residential labels. Inputs can form unseen combinations. This model does not replace a professional valuation.')
