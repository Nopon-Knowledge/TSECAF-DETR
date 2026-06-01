# Implementation Directory

This directory contains the retained PyTorch implementation for
TSECAF-DETR wheat-head detection. Recommended command presets use the
`tsecaf_detr_*` naming; legacy aliases remain available for older experiment
logs.

Use the repository-level README for the full installation, dataset, training,
evaluation, visualization, export, and profiling instructions.

Key entry points:

- `configs/rtdetrv2/`: baseline, TSECAF-DETR, pathway-ablation, and direct-transfer configs.
- `src/zoo/rtdetr/hybrid_encoder.py`: context-detail evidence formation.
- `src/zoo/rtdetr/rtdetrv2_decoder.py`: compact-query allocation.
- `tools/train_gwhd_a40.py`: training presets.
- `tools/test_gwhd_a40.py`: GWHD2021 and cross-dataset evaluation presets.
- `tools/prepare_wheat_datasets.py`: dataset conversion helpers.

Run commands from this directory:

```bash
python tools/train_gwhd_a40.py --preset tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset wheat_ears_tsecaf_detr_r50
```
