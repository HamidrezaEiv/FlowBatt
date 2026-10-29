# Prepared datasets

The five NumPy archives provide shared inputs for trajectory models, baselines, and CNN/XAI-CNN experiments. Load them with `numpy.load(path, allow_pickle=False)`. They contain numeric arrays only. The measurements originate from the original public battery datasets—MATR, HUST, CALCE, HNEI, RWTH, SNL, and UL_PUR. 

| Dataset | Train / test cells | Capacity matrix per cell | SOH target points |
| --- | ---: | --- | ---: |
| MATR-1 | 41 / 42 | 100 × 100 | 256 |
| MATR-2 | 41 / 40 | 100 × 100 | 256 |
| HUST | 55 / 22 | 100 × 100 | 256 |
| MIX-20 | 256 / 147 | 20 × 100 | 256 |
| MIX-100 | 205 / 137 | 100 × 100 | 256 |


## Shared array keys

| Keys | Meaning |
| --- | --- |
| `x_train`, `x_test` | Conditioning matrices in fixed split and row order. |
| `y_train_scaled`, `y_test_scaled` | Scaled, preprocessed 256-point SOH targets. |
| `y_train_physical`, `y_test_physical` | Inverse-transformed SOH fractions. |
| `label_min`, `label_max` | Scalar training-set SOH transform parameters. |
| `rul_train_scaled`, `rul_test_scaled` | Regular cached scalar RUL labels in stored log/Z-score coordinates; fresh CNN/XAI-CNN training uses `rul_train_scaled`. |
| `rul_train_legacy`, `rul_test_legacy` | Regular scalar labels in cycles, used as CNN/XAI-CNN metric references. |
| `rul_log_base`, `rul_zscore_mean`, `rul_zscore_std` | Cached scalar-label transform parameters. |


The files preserve the fixed split and row order shown above.

