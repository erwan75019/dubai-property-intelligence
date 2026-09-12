# V2 pipeline and data audit

## V1 findings

V1 explicitly whitelisted actual surface, date, area, subtype, rooms, completion status, tenure and project. No target, unit-price diagnostic, area ratio, transaction ID or procedure-area predictor entered training. Its feature builder discarded extra source columns; V2 instead rejects them explicitly. Numeric scaling and category grouping lived inside sklearn pipelines and were fitted separately for random and temporal validation. PROJECT_EN used a maximum of 80 one-hot categories, losing much of its long tail.

V1 temporal training ended June 11; validation ran June 12–July 22; refit used 79,891 records through July 22; holdout ran July 23–September 11 (18,097 records). Random validation came only from pre-holdout development data. No learned transform was fitted to holdout data. Existing whole-snapshot EDA means later external validation is still needed.

## Preserved and revised eligibility rules

Residential usage + Sales group + the five requested subtypes defines the population. Sale, Sell - Pre registration, Delayed Sell and Sale On Payment Plan remain a conservative working allowlist. Development and lease-to-own procedures are different or uncertain economic events; this is a modeling assumption, not an authoritative determination of individual-sale status.

Positive finite price and actual area and valid date are essential. Known procedure/actual-area ratios outside [0.5, 2] remain excluded as possible scope mismatches. The ratio rule may remove legitimate entries; no new threshold was selected from holdout performance. Missing procedure area is allowed because actual area is the predictor.

V2 adds rejection of Office/Shop room labels attached to residential subtypes (one February Flat entry), and checks duplicate IDs for conflicting input attributes as well as price. There are no conflicting IDs in this export. Consistent repeated IDs account for 19 excluded entries. The final holdout population is unchanged from V1.

No luxury price cap, IQR removal, unit-price trimming or target-quantile selection is applied. Development-only diagnostics flag 1 entry below AED 100/m² and 40 above AED 100,000/m² for review. These broad thresholds are not validity tests. The official database can correctly represent events that are unsuitable for individual-home regression.

## Feature availability and leakage contract

| Field | V2 treatment | Reason |
|---|---|---|
| ACTUAL_AREA | Input plus log area | Known surface; require positive finite m² |
| INSTANCE_DATE | Input → year/month/quarter | Date of intended transaction is supplied by the user |
| AREA_EN, PROJECT_EN | Inputs | Location/project identity available before pricing; unseen strings accepted |
| PROP_SB_TYPE_EN, ROOMS_EN | Inputs | Property characteristics; missing optional rooms become Unknown |
| IS_OFFPLAN_EN, IS_FREE_HOLD_EN | Inputs | Known status/tenure, with optional Unknown |
| PROCEDURE_EN | Eligibility only | Exact eventual registration procedure not reliably available to ordinary app users |
| TRANS_VALUE | Target only | Unknown outcome |
| PRICE_PER_SQM | Internal alternate target / audit only | Derived from the outcome; never an input |
| AREA_RATIO, PROCEDURE_AREA | Audit/filter only | Procedure scope may differ from unit scope |
| TRANSACTION_NUMBER | Deduplication only | Identifier, not stable property characteristics |
| PARKING | Excluded | Counts mixed with bay labels |
| GROUP_EN, USAGE_EN, buyer/seller counts | Excluded | Constant within this snapshot/cohort |
| MASTER_PROJECT_EN | Excluded | Almost entirely missing |
| Nearest metro/mall/landmark | Excluded | Omitted from this bounded V2 feature experiment; may add location context later |

An explicit raw-input allowlist rejects extra and target-derived fields; no unsafe target encoding is performed. Unit-price training uses y divided by surface inside a wrapper and reconstructs total AED in the same serialized prediction path. CatBoost's categorical processing is confined to each training fit, with date-ordered input. One-hot vocabulary and scaling are training-local for other estimators.

## Validation safeguards and limitations

Chronological training, validation, calibration and final comparison have separate fixed dates. Tuning folds end before the external validation period. Final estimator fitting ends July 8; calibration is July 9–22. The final comparison starts July 23. Segment models use identical cutoffs and are compared against the global model on identical segment rows. No final-test score influences estimator or routing selection. Holdout subgroup inspection does not trigger retrospective cohort changes.

The V1 holdout is previously examined data, so V2 is a frozen comparative experiment, not an untouched external test. Price-dependent record consistency checks and exploratory scope rules should be repeated with documented provenance on a new extract. Stable unit IDs are absent; repeat-property overlap remains possible. Final selection prioritizes validation MAE with RMSE as a tie-break, not a multiobjective optimizer; tuning's MAE/RMSE tradeoff is reported transparently. Two CV folds do not establish long-term stability.
