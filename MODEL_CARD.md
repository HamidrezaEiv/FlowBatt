# FlowBatt model card

## Intended use

FlowBatt supports research on battery SOH trajectory prediction, derived RUL estimation, empirical prediction intervals, and synthetic degradation trajectories. The checkpoints are not validated for independent safety decisions or deployment on unseen chemistries or operating regimes.

## Inputs and outputs

The conditioning input is a baseline-subtracted early-cycle discharge-capacity matrix. Models predict 256-point normalized SOH trajectories; evaluation inverts the stored training-set transform and interpolates to a 2560-cycle grid.

Five benchmark settings are included: MATR-1, MATR-2, HUST, MIX-20, and MIX-100. Their splits, preprocessing, and target definitions are documented in [Datasets](datasets/README.md).

## Models

FlowBatt and DiffBatt share a compact DiT-style backbone and LeakyReLU capacity encoder. The supervised Transformer uses its own corresponding architecture. Prediction FlowBatt checkpoints use conditioning dropout 0.0; the separate MATR-1 synthesis checkpoint uses 0.2 for classifier-free guidance. 

The [CNN/XAI-CNN experiments](docs/experiments.md#scalar-cnn-and-xai-cnn) include ten original MATR-1 XAI-CNN checkpoints and training scripts for both activations on all five datasets. New runs use stored regular RUL labels and native capacity matrices: 100 × 100, or 20 × 100 for MIX-20. These models predict scalar RUL, not SOH trajectories.

RF, MLP, Quantile RF, and Quantile MLP predict 256-point SOH trajectories. Their input representations and training settings are described in the [experiment guide](docs/experiments.md#trajectory-and-quantile-baselines).

## Evaluation

Trajectory SOH evaluation reports the mean of per-battery RMSEs across the full interpolated 2560-cycle horizon, after clipping reference and prediction from below at EOL, in SOH percentage points.

## Licensing

See [LICENSE](LICENSE) for implementation terms and [Datasets](datasets/README.md) for data attribution. The source-code license does not grant additional rights to datasets or checkpoints.
