# Experiments

Run these commands from the repository root after installation. The dataset names are `MATR-1`, `MATR-2`, `HUST`, `MIX-20`, and `MIX-100`. Inputs come from `datasets/` and `checkpoints/`; commands write new results under `workspaces/`.

## SOH and RUL prediction

```bash
python -m flowbatt.evaluate --dataset MATR-1 --model flowbatt --num-samples 10
python -m flowbatt.evaluate --dataset MATR-1 --model diffbatt --num-samples 10
python -m flowbatt.evaluate --dataset MATR-1 --model transformer --num-samples 10
```

Each command saves trajectories and metrics in `workspaces/evaluation/MATR-1/{model}/`. Use `--output` for a separate run. FlowBatt and DiffBatt draw multiple trajectories from one checkpoint; Transformer uses independently trained checkpoints. 

### Evaluate newly trained models

Training saves final-epoch weights to `workspaces/training/<model>/<dataset>/<model>_seed_<seed>.pth`, unless `--output` is supplied. Conditioning dropout remains configurable with `--conditioning-dropout` and is recorded in the accompanying JSON. Pass the saved file to the evaluator with the same model, dataset, and architecture configuration used for training:

```bash
python -m flowbatt.evaluate --model flowbatt --dataset MATR-1 \
  --checkpoint workspaces/training/flowbatt/MATR-1/flowbatt_seed_0.pth \
  --num-samples 10 --output workspaces/evaluation/MATR-1/flowbatt_retrained

python -m flowbatt.evaluate --model diffbatt --dataset MATR-1 \
  --checkpoint workspaces/training/diffbatt/MATR-1/diffbatt_seed_0.pth \
  --num-samples 10 --output workspaces/evaluation/MATR-1/diffbatt_retrained

python -m flowbatt.evaluate --model transformer --dataset MATR-1 \
  --checkpoint workspaces/training/transformer/MATR-1/transformer_seed_0.pth \
  --output workspaces/evaluation/MATR-1/transformer_retrained_seed_0
```

To evaluate two independently trained Transformer seeds, first train seed 1 in addition to seed 0, then supply both files:

```bash
python -m flowbatt.train --model transformer --dataset MATR-1 --seed 1 --device cuda
python -m flowbatt.evaluate --model transformer --dataset MATR-1 \
  --checkpoint workspaces/training/transformer/MATR-1/transformer_seed_0.pth \
  --checkpoint workspaces/training/transformer/MATR-1/transformer_seed_1.pth \
  --output workspaces/evaluation/MATR-1/transformer_retrained_seeds_0_1
```

## Prediction intervals

```bash
python -m flowbatt.experiments.uncertainty \
  --models flowbatt diffbatt --num-samples 10 --sample-chunk 10 \
  --diffusion-steps 1000 --seed 2026 --device cuda
```

The default experiment covers all five datasets and writes to `workspaces/generative_uq/`. Empirical 90% prediction intervals use the 5th and 95th sample quantiles. Coverage and width are evaluated on the full interpolated, EOL-clipped horizon; width is in SOH percentage points. 

FlowBatt uses adaptive Dopri5 integration with tolerances `atol=rtol=1e-5`. DiffBatt uses the full 1000-step reverse process in this comparison.

## Trajectory and quantile baselines

```bash
python -m flowbatt.experiments.full_matrix_baselines --models rf mlp --seeds 10
python -m flowbatt.experiments.quantile_baselines \
  --models qrf qmlp --seeds 10 --device cuda
python -m flowbatt.experiments.plot_quantiles --results-root workspaces
```

RF, MLP, and Quantile MLP use full capacity matrices. Quantile RF uses the final observed early-life discharge-capacity difference curve at the native 100 voltage points, without resizing. All four use training-only input and SOH-target Z-score transforms and report mean per-battery trajectory RMSE.

Quantile RF uses a 100-tree forest by default and estimates conditional 5th, 50th, and 95th percentiles from training-observation leaf weights. Quantile MLP uses ordered quantile heads and pinball loss. Its frozen settings are in `configs/matched_quantile.json`. To use newly selected deterministic MLP settings instead, pass `--baseline-root workspaces/baselines`.

## Guided generation and RF augmentation

Standard prediction uses conditioning dropout 0.0 and guidance scale 1.0. The separate MATR-1 synthesis checkpoint uses conditioning dropout 0.2:

```bash
python -m flowbatt.evaluate --model flowbatt --dataset MATR-1 \
  --conditioning-dropout 0.2 --guidance-scale 2.0 --num-samples 10 \
  --output workspaces/guided_example

python -m flowbatt.experiments.augmentation --augment-flowbatt --dataset MATR-1 \
  --synthetic-per-sample 5 --flow-device cuda \
  --results-dir workspaces/rf_synthesis
```

The RF experiment compares real-only training with augmentation at guidance scales 0.5, 1, 2, 3, and 4. 


## Scalar CNN and XAI-CNN

```bash
python xaicnn/evaluate.py
python xaicnn/train.py --model xai-cnn --dataset MIX-20 --check-data
python -u xaicnn/train.py --model xai-cnn --dataset MATR-1 --device auto
python xaicnn/evaluate.py --run workspaces/xai_cnn_MATR-1

python -u xaicnn/train.py --model cnn --dataset MIX-20 --seed 0 --device auto
python xaicnn/evaluate.py --run workspaces/cnn_MIX-20
```

The root installation covers these scripts. Both models can be trained on `MATR-1`, `MATR-2`, `HUST`, `MIX-20`, or `MIX-100`. 
