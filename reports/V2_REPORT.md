# V2 experiment report

## Decision and comparison limits

The selected artifact is a CatBoost price-per-square-metre model with separate Flat and Villa specialists and a global fallback for the broader Residential labels. All targets are evaluated as total price in AED. Validation selected the model and routing before any final holdout prediction. Its settings were frozen in `v2_selection.json`.

The V1 holdout was already inspected. V2 preserves it as a chronological, same-record comparison; it is not an independently untouched external test. Final fitting stops July 8 to reserve calibration, whereas V1 used records through July 22. The only new semantic exclusion is one February Flat/Office mismatch, leaving all 18,097 holdout records unchanged.

## Final results

Segment-only rows are evaluated on different populations and cannot be compared to the global headline as though they were the same task.


| model | formulation | target | split_strategy | rows | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V1 hist_gradient_boosting | global | direct | same chronological future holdout | 18097 | 360,390.627 | 2,816,302.907 | 0.430 |
| hist_gradient_boosting | global | direct | chronological future holdout | 18097 | 368,054.566 | 2,643,654.656 | 0.498 |
| hist_gradient_boosting | global | log | chronological future holdout | 18097 | 321,747.526 | 2,635,767.715 | 0.501 |
| catboost | global | per_sqm | chronological future holdout | 18097 | 305,127.032 | 2,171,196.478 | 0.661 |
| catboost | Flat-only | per_sqm | same future dates, segment subset | 16998 | 248,241.301 | 1,035,303.151 | 0.809 |
| catboost | Villa-only | per_sqm | same future dates, segment subset | 1075 | 751,734.690 | 6,101,899.014 | 0.436 |
| segmented ensemble | routed | per_sqm | chronological future holdout | 18097 | 294,289.130 | 2,239,690.895 | 0.639 |


Selected V2 MAE improves by **18.34%** and RMSE by **20.47%** versus V1. R² rises from 0.430 to 0.639. The global unit-price model has higher R² and lower RMSE than the selected routed model, but higher MAE. We do not replace the preselected model after inspecting this table.

Paired resampling of calendar days gives a V1-minus-V2 MAE reduction of AED 66,101, with approximate 95% bounds AED 48,584–85,732. This supports a meaningful improvement for this snapshot. Dependence across dates, repeat properties, prior EDA and limited history prevent stronger external claims.

## Full validation benchmark

Validation dates: June 12–July 8; initial training through June 11. Each entry below uses the same validation population. The two strongest families were tuned using only earlier training-period CV. All values are original-AED errors, including log and unit-price targets.


| model | formulation | stage | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| catboost | per_sqm | tuned | 259,674.400 | 1,683,496.102 | 0.777 |
| catboost | per_sqm | benchmark | 261,373.612 | 1,388,416.017 | 0.849 |
| hist_gradient_boosting | per_sqm | tuned | 267,696.394 | 1,532,903.983 | 0.815 |
| hist_gradient_boosting | per_sqm | benchmark | 278,419.544 | 1,553,279.475 | 0.810 |
| hist_gradient_boosting | log | benchmark | 296,253.489 | 2,062,220.136 | 0.666 |
| catboost | log | benchmark | 307,058.988 | 2,072,860.292 | 0.662 |
| hist_gradient_boosting | direct | benchmark | 335,774.609 | 1,956,290.772 | 0.699 |
| catboost | direct | benchmark | 337,965.988 | 2,097,110.603 | 0.654 |
| random_forest | log | benchmark | 339,628.686 | 1,387,909.940 | 0.849 |
| ridge | log | benchmark | 353,610.333 | 2,044,977.537 | 0.671 |
| ridge | per_sqm | benchmark | 365,228.393 | 1,988,632.735 | 0.689 |
| random_forest | direct | benchmark | 374,837.570 | 1,593,193.922 | 0.801 |
| random_forest | per_sqm | benchmark | 395,715.083 | 1,474,623.233 | 0.829 |
| ridge | direct | benchmark | 561,718.567 | 2,142,534.899 | 0.639 |
| dummy | per_sqm | benchmark | 576,899.839 | 2,570,400.616 | 0.481 |
| dummy | log | benchmark | 1,104,834.396 | 3,579,402.655 | -0.007 |
| dummy | direct | benchmark | 1,104,834.396 | 3,579,402.655 | -0.007 |


Tuned CatBoost uses MAE loss, 650 iterations, depth 8, learning rate 0.04 and L2 regularization 10. Its MAE gain over untuned CatBoost is small and comes with worse validation RMSE. The predeclared ranking prioritizes MAE with RMSE as a tie-break; this is a visible tradeoff, not a claim that tuning dominates every metric. A future experiment could predeclare a stronger multiobjective or uncertainty-aware selection policy, but should not derive it from this holdout.

## Segment analysis on matching records


| segment | scope | rows | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| Flat | global | 16998 | 262,151.356 | 1,072,745.846 | 0.795 |
| Flat | specialist | 16998 | 248,241.301 | 1,035,303.151 | 0.809 |
| Villa | global | 1075 | 714,237.390 | 5,558,693.432 | 0.532 |
| Villa | specialist | 1075 | 751,734.690 | 6,101,899.014 | 0.436 |


Flat specialization improves holdout MAE by 5.3% and RMSE by 3.5% versus global CatBoost on the same Flat records. Villa specialization worsens MAE by 5.2% and RMSE by 9.8%. Both passed the validation gate; the Villa benefit failed to generalize. Keep this failure explicit. The combined route still reduces overall MAE relative to the global model, but increases RMSE. Do not use final-test inspection to silently disable a route.

## Feature investigation

Project and log-area ablations use the selected global model configuration and validation period only. No ablation retrains against holdout labels.


| features | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| no_log_area | 260,119.388 | 1,643,393.222 | 0.788 |
| no_project | 297,050.401 | 1,678,122.605 | 0.779 |
| full | 259,674.400 | 1,683,496.102 | 0.777 |


Project removal raises validation MAE materially; log area has a small MAE effect. Both log area and project remain in the predeclared input specification. Month and quarter are deterministic date features. Property subtype, rooms, surface, status and tenure remain inputs. Registration procedure is excluded because its availability for an ordinary pre-sale estimate is not assured.

## Chronological cross-validation stability

The following are the randomized searches. Scores in sklearn CSVs are negative errors; this table reports positive MAE/RMSE. Two folds reveal some time variation but do not establish long-term stability.


### v2_cv_catboost

| parameters | mean_CV_MAE_AED | std_CV_MAE_AED | mean_CV_RMSE_AED | rank |
| --- | --- | --- | --- | --- |
| {'pipeline__regressor__loss_function': 'MAE', 'pipeline__regressor__learning_rate': 0.04, 'pipeline__regressor__l2_leaf_reg': 10, 'pipeline__regressor__iterations': 650, 'pipeline__regressor__depth': 8} | 343,130.905 | 18,551.010 | 1,957,809.609 | 1 |
| {'pipeline__regressor__loss_function': 'MAE', 'pipeline__regressor__learning_rate': 0.08, 'pipeline__regressor__l2_leaf_reg': 10, 'pipeline__regressor__iterations': 650, 'pipeline__regressor__depth': 6} | 346,835.945 | 23,955.285 | 2,036,116.342 | 2 |
| {'pipeline__regressor__loss_function': 'RMSE', 'pipeline__regressor__learning_rate': 0.04, 'pipeline__regressor__l2_leaf_reg': 3, 'pipeline__regressor__iterations': 650, 'pipeline__regressor__depth': 8} | 360,228.221 | 174.602 | 1,972,366.273 | 3 |

### v2_cv_hist_gradient_boosting

| parameters | mean_CV_MAE_AED | std_CV_MAE_AED | mean_CV_RMSE_AED | rank |
| --- | --- | --- | --- | --- |
| {'pipeline__regressor__max_leaf_nodes': 15, 'pipeline__regressor__max_iter': 350, 'pipeline__regressor__loss': 'squared_error', 'pipeline__regressor__learning_rate': 0.05, 'pipeline__regressor__l2_regularization': 20} | 433,696.578 | 30,989.962 | 2,514,077.541 | 3 |
| {'pipeline__regressor__max_leaf_nodes': 31, 'pipeline__regressor__max_iter': 350, 'pipeline__regressor__loss': 'squared_error', 'pipeline__regressor__learning_rate': 0.1, 'pipeline__regressor__l2_regularization': 20} | 368,786.559 | 16,307.060 | 2,030,308.589 | 1 |
| {'pipeline__regressor__max_leaf_nodes': 63, 'pipeline__regressor__max_iter': 200, 'pipeline__regressor__loss': 'squared_error', 'pipeline__regressor__learning_rate': 0.05, 'pipeline__regressor__l2_regularization': 20} | 387,721.289 | 25,103.463 | 2,249,885.982 | 2 |


## Uncertainty and failure analysis

Calibration uses 5,222 records from July 9–22, after final estimator fitting. Absolute log residuals define a multiplicative range with a finite-sample rank correction. Nominal coverage is 90%; observed holdout coverage is **88.27%** and median width is AED 555,764. There is no coverage guarantee under temporal shift. In particular, observed prices above AED 10M achieve only 40.9% coverage.

The tables below are post-selection diagnostics. Price bands use targets only for analysis and never enter features. Bias is prediction minus observed price; negative bias indicates underprediction. Small-group and narrow-band R² can be unstable or negative.


### PROP_SB_TYPE_EN

| group | rows | MAE | RMSE | R2 | bias_AED | interval_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Flat | 16998 | 248,241.301 | 1,035,303.151 | 0.809 | -44,464.465 | 0.886 |
| Villa | 1075 | 751,734.690 | 6,101,899.014 | 0.436 | -349,932.845 | 0.846 |
| Residential Flats | 12 | 21,999,902.621 | 51,307,261.485 | 0.379 | -18,905,996.362 | 0.417 |
| Residential / Attached Villas | 9 | 164,467.711 | 242,351.396 | 0.618 | -58,324.924 | 0.778 |
| Residential / Villas | 3 | 10,850,306.873 | 17,735,920.278 | -1.855 | -10,850,306.873 | 0.667 |

### price_band

| group | rows | MAE | RMSE | R2 | bias_AED | interval_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| <1M | 7921 | 118,687.300 | 371,832.215 | -3.014 | 71,415.993 | 0.887 |
| 1–2M | 5721 | 142,574.900 | 218,120.163 | 0.318 | -240.469 | 0.919 |
| 2–5M | 3808 | 370,694.507 | 534,297.349 | 0.547 | -148,108.643 | 0.865 |
| 5–10M | 466 | 1,178,007.692 | 1,642,215.519 | -0.772 | -664,632.846 | 0.687 |
| 10M+ | 181 | 8,891,718.341 | 21,932,504.035 | 0.369 | -5,980,808.431 | 0.409 |

### ROOMS_EN

| group | rows | MAE | RMSE | R2 | bias_AED | interval_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Studio | 6499 | 61,483.072 | 121,376.738 | 0.594 | 3,402.068 | 0.946 |
| 1 B/R | 6424 | 193,106.787 | 365,873.641 | 0.642 | -3,280.491 | 0.862 |
| 2 B/R | 3281 | 404,689.297 | 1,214,887.352 | 0.673 | -110,857.963 | 0.842 |
| 3 B/R | 1150 | 836,890.489 | 2,241,172.530 | 0.766 | -350,193.936 | 0.807 |
| 4 B/R | 529 | 1,203,664.067 | 3,462,808.256 | 0.772 | -268,262.286 | 0.849 |
| 5 B/R | 161 | 1,046,000.554 | 1,828,013.024 | 0.734 | -401,298.151 | 0.733 |
| nan | 42 | 8,198,862.154 | 27,911,770.547 | 0.499 | -5,767,949.645 | 0.571 |
| PENTHOUSE | 7 | 940,453.784 | 1,685,324.927 | -5.422 | 885,771.653 | 0.714 |
| 6 B/R | 4 | 59,841,123.887 | 99,476,037.716 | 0.015 | -45,986,878.532 | 0.250 |

### IS_OFFPLAN_EN

| group | rows | MAE | RMSE | R2 | bias_AED | interval_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Off-Plan | 13271 | 219,641.542 | 1,051,108.843 | 0.824 | -91,964.149 | 0.925 |
| Ready | 4826 | 499,562.264 | 3,971,412.873 | 0.546 | -35,531.778 | 0.768 |

### Highest-error areas with at least 100 holdout entries

| group | rows | MAE | RMSE | R2 | bias_AED | interval_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| DUBAI HILLS | 176 | 1,676,224.781 | 13,218,100.685 | 0.552 | -780,507.104 | 0.750 |
| BURJ KHALIFA | 262 | 1,152,348.683 | 2,574,445.198 | 0.732 | -188,268.734 | 0.668 |
| Palm Jabal Ali | 132 | 987,798.773 | 1,387,866.442 | 0.798 | -287,136.432 | 0.689 |
| DUBAI MARINA | 335 | 873,441.012 | 1,521,352.873 | 0.576 | -83,682.267 | 0.436 |
| BUSINESS BAY | 531 | 825,176.592 | 3,084,379.127 | 0.723 | -456,332.023 | 0.819 |
| Wadi Al Safa 3 | 105 | 714,781.208 | 1,031,184.741 | 0.906 | 299,771.280 | 0.543 |
| Hadaeq Sheikh Mohammed Bin Rashid | 115 | 686,810.851 | 2,467,375.717 | 0.842 | -95,739.112 | 0.913 |
| JUMEIRAH LAKES TOWERS | 167 | 431,441.435 | 1,216,498.837 | 0.484 | 254,063.254 | 0.749 |
| DUBAI MARITIME CITY | 105 | 409,152.045 | 664,764.750 | 0.706 | -312,745.507 | 0.914 |
| Palm Deira | 177 | 404,646.106 | 855,195.024 | 0.808 | 6,060.521 | 0.932 |


The worst errors concentrate in luxury transactions and rare broad Residential labels. Residential Flats MAE remains roughly AED 22 million with only 12 entries, suggesting unresolved scope or extreme-tail problems. Major room groups and statuses can also differ; aggregate improvement is not uniform reliability.

## Interpretation and deployment

The app uses the exact artifact, including target inversion and routing. Permutation importance measures total-AED sensitivity of that complete prediction path. SHAP explains a routed branch in its internal units (AED/m²), with a reconstruction assertion; it does not establish causal effects. For a unit-price model, surface permutation affects both learned unit price and total-price multiplication.

No price cap, IQR trim or unsafe target encoding was introduced. Domain review of bulk/partial-interest sales, stable property IDs and a newer validation extract remain the most important next steps. Floor, view, condition and ownership share are missing. The V1 source/artifact and aggregate reports are archived locally so the original result is traceable.
