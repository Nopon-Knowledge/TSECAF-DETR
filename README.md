# TSECAF-DETR

PyTorch implementation of **TSECAF-DETR: Preserving Candidate Coverage under Fixed Query Budgets for Dense Wheat Head Detection**.

TSECAF-DETR is built on RT-DETRv2. It addresses the loss of localized target coverage as encoder proposals are reduced to a fixed decoder query budget. Image-conditioned context-detail exchange shapes the candidate pool; a compact anchor prior and area-penalized ranking select proposals for decoder initialization. The decoder and training objective remain those of RT-DETRv2.

> Public commands below use the TSECAF-DETR preset names. Legacy preset and config aliases are retained for older experiment logs.

## Overview

<p align="center">
  <img src="assets/readme/architecture.png" alt="TSECAF-DETR architecture" width="900">
</p>

Candidate-coverage pathway:

- **Context-detail evidence formation**: token statistics guide semantic-detail exchange at the finest top-down fusion stage.
- **Detail-sensitive low-level evidence**: boundary responses are preserved before semantic information is exchanged across branches.
- **Geometric selection**: the anchor prior uses `g=0.04`; objectness logits are ranked by `s - lambda * (w*h)` with `lambda=8` in the wheat setting.
- **Real-time end-to-end inference**: the detector preserves the RT-DETRv2 decoding path without NMS.

## Results Reported in the Current Manuscript

Protocol A detector comparison on GWHD2021 (`1024 x 1024`, AP100):

| Model | AP | AP50 | AP75 | AP-S | FPS |
|---|---:|---:|---:|---:|---:|
| YOLO11m | 50.71 | 91.58 | 56.84 | 47.91 | 145.2 |
| RT-DETRv2-R50 | 50.68 | 91.74 | 56.92 | 47.96 | 108.5 |
| **TSECAF-DETR-R50** | **53.62** | **93.18** | **60.41** | **51.46** | 102.8 |

Held-out GWHD2021 test results on protocol B and direct transfer:

| Evaluation | RT-DETRv2-R50 | TSECAF-DETR-R50 | Metric |
|---|---:|---:|---|
| GWHD2021 held-out test | 49.82 | **52.45** | AP100 |
| GWHD2021 held-out test | 7.439 | **5.866** | Counting MAE |
| Wheat Ears direct transfer (48 images) | 39.62 | **43.81** | AP300 |
| SKU-110K independently trained | 43.28 | **44.92** | AP100 |

At the fixed-budget comparison, 300-query TSECAF-DETR reaches 52.91 AP300 at 9.24 ms, compared with 50.75 AP300 at 10.12 ms for the 700-query baseline. At 500 queries, maximum-matching target coverage at IoU 0.75 increases from 60.8% to 69.4%.

<p align="center">
  <img src="assets/readme/accuracy_efficiency.png" alt="Accuracy-efficiency trade-off" width="720">
</p>

Qualitative detections on dense GWHD2021 validation images:

|  |  |
|---|---|
| <img src="assets/readme/case_01.png" width="420"> | <img src="assets/readme/case_02.png" width="420"> |
| <img src="assets/readme/case_03.png" width="420"> | <img src="assets/readme/case_04.png" width="420"> |

## Repository Layout

```text
tsecaf_detr/
  configs/
    dataset/      # COCO-style wheat dataset configs
    rtdetrv2/     # baseline, TSECAF-DETR, transfer, and pathway-ablation configs
  src/
    zoo/rtdetr/   # RT-DETRv2 model and candidate-coverage pathway
  tools/
    train_gwhd_a40.py
    test_gwhd_a40.py
    prepare_wheat_datasets.py
    visualize_gwhd_predictions.py
    run_profile.py
```

## Installation

```bash
cd tsecaf_detr
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the `torch` and `torchvision` wheels that match your CUDA runtime before installing the remaining dependencies if needed.

The paper experiments were run on a single NVIDIA A40 GPU. Some environments may need cuDNN disabled for stable RT-DETRv2 training:

```bash
export RTDETR_DISABLE_CUDNN=1
```

## Datasets

Datasets are not included. Put COCO-style datasets under `tsecaf_detr/datasets` or set:

```bash
export WHEAT_DATASETS_ROOT=/path/to/datasets
```

Expected layout:

```text
datasets/
  gwhd_2021/
    train2017/
    val2017/
    annotations/instances_train2017.json
    annotations/instances_val2017.json
  Wheat_Ears_Detection_Dataset_rtdetr/
    train2017/
    val2017/
    annotations/instances_train2017.json
    annotations/instances_val2017.json
  global-wheat-codalab-official_rtdetr/
    train2017/
    val2017/
    annotations/instances_train2017.json
    annotations/instances_val2017.json
```

Dataset protocol:

- Protocol A: random GWHD2021 partition with 5499 training and 1016 validation images; used for controlled comparisons and mechanism analysis.
- Protocol B: random GWHD2021 partition with 4399 training, 1100 development, and 1016 held-out test images.
- Wheat Ears transfer: 48-image evaluation subset (6240 annotations), evaluated directly with protocol B checkpoints and no target-domain fine-tuning.
- SKU-110K: independent task training on the official split; wheat allocation settings are retained for the fixed-setting comparison.

Conversion helpers:

```bash
python tools/prepare_wheat_datasets.py --dataset wheat_ears
python tools/prepare_wheat_datasets.py --dataset global_wheat_codalab
```

The Global Wheat CodaLab conversion utility remains available for project experiments; those results are not part of the current manuscript.

## Training

Run commands from `tsecaf_detr/`.

Train the final TSECAF-DETR-R50 model:

```bash
python tools/train_gwhd_a40.py --preset tsecaf_detr_r50
```

Train paired RT-DETRv2 baselines:

```bash
python tools/train_gwhd_a40.py --preset baseline_r18
python tools/train_gwhd_a40.py --preset baseline_r34
python tools/train_gwhd_a40.py --preset baseline_r50
```

Train TSECAF-DETR model scales:

```bash
python tools/train_gwhd_a40.py --preset tsecaf_detr_r18
python tools/train_gwhd_a40.py --preset tsecaf_detr_r34
python tools/train_gwhd_a40.py --preset tsecaf_detr_r50
```

Outputs are written to `output/`, which is ignored by Git.

## Evaluation

GWHD2021 validation:

```bash
python tools/test_gwhd_a40.py --preset tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset baseline_r50
```

Direct transfer without target-domain fine-tuning:

```bash
python tools/test_gwhd_a40.py --preset wheat_ears_tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset wheat_ears_baseline_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_baseline_r50
```

Profile parameters and FLOPs:

```bash
python tools/run_profile.py \
  -c configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml
```

Visualize predictions:

```bash
python tools/visualize_gwhd_predictions.py --preset tsecaf_detr_r50_val_best
```

## Paper-to-Code Map

| Paper item | Code location |
|---|---|
| Context-detail evidence formation | `src/zoo/rtdetr/hybrid_encoder.py` |
| Compact-query allocation | `src/zoo/rtdetr/rtdetrv2_decoder.py` |
| TSECAF-DETR configs | `configs/rtdetrv2/tsecaf_detr_*_gwhd.yml` |
| RT-DETRv2 baseline configs | `configs/rtdetrv2/rtdetrv2_*_gwhd_baseline.yml` |
| Cross-dataset transfer configs | `configs/rtdetrv2/*eval_wheat_ears*.yml`, `configs/rtdetrv2/*eval_global_wheat_codalab*.yml` |
| Training presets | `tools/train_gwhd_a40.py` |
| Evaluation presets | `tools/test_gwhd_a40.py` |

## Notes

- Public datasets and official RT-DETRv2 pretrained checkpoints must be downloaded separately.
- External baselines such as Faster R-CNN, RetinaNet, YOLO, D-FINE, and LW-DETR are not vendored in this reduced release.
- Generated checkpoints, logs, exported models, and visualization outputs are ignored by Git.
- For exact table reproduction, archive the trained checkpoints and per-seed evaluation outputs with the manuscript version.

## License

This code is released under the license in [LICENSE](LICENSE).

## Citation

If this repository is useful for your work, please cite the associated TSECAF-DETR paper after publication, together with RT-DETRv2 and the public wheat datasets used in your experiments.
