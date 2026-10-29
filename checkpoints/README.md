# Pretrained checkpoints

The 61 trajectory-model checkpoints are PyTorch tensor state dictionaries. Load with `torch.load(path, map_location=device, weights_only=True)`, then use the matching model definition.

| Directory | Models |
| --- | --- |
| `flowbatt/` | One prediction checkpoint per dataset; conditioning dropout 0.0. |
| `flowbatt_synthesis/` | One MATR-1 synthesis checkpoint; conditioning dropout 0.2. |
| `diffbatt/` | One prediction checkpoint per dataset. |
| `transformer/{DATASET}/` | Ten independently trained checkpoints, `seed_0.pth` through `seed_9.pth`. |

