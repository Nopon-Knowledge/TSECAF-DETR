<a id="top"></a>
<div align="center">

# 🌾 TSECAF-DETR

### Preserving Candidate Coverage under Fixed Query Budgets<br>for Dense Wheat Head Detection

**Image-conditioned candidate formation · Geometric query selection · End-to-end detection**

[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](tsecaf_detr/)
[![Task](https://img.shields.io/badge/Task-Dense_Object_Detection-477A5B?style=flat-square)](#overview)
[![Backbone](https://img.shields.io/badge/Baseline-RT--DETRv2-5267A5?style=flat-square)](#results)
[![License](https://img.shields.io/badge/License-Repository_License-7A7A7A?style=flat-square)](LICENSE)

[Overview](#overview) · [Results](#results) · [Quick Start](#quick-start) · [Datasets](#datasets) · [Code Map](#paper-to-code)

</div>

---

TSECAF-DETR preserves localized target coverage when encoder proposals are reduced to a fixed decoder query budget. Statistical context–detail exchange improves the candidate pool, while a compact anchor prior and area-penalized ranking select candidates for the RT-DETRv2 decoder.

| **69.4% selected coverage** | **52.45 held-out AP100** | **52.91 AP300 with 300 queries** |
|:---:|:---:|:---:|
| vs. 60.8% baseline at IoU 0.75 · Protocol A | vs. 49.82 baseline · Protocol B | vs. 50.75 with 700 baseline queries · Protocol A |

<a id="overview"></a>
## Overview

<p align="center">
  <img src="assets/readme/architecture.png" alt="TSECAF-DETR architecture: context-detail exchange followed by geometric proposal selection" width="960">
</p>

| Candidate formation | Fixed-budget selection |
|---|---|
| Prepare local detail with parallel depthwise convolutions. | Initialize proposal geometry with a compact anchor prior. |
| Use image-wise means and variances to guide bidirectional context–detail exchange. | Rank objectness logits with a normalized box-area penalty. |
| Apply the exchange at the finest top-down fusion stage. | Refine the selected proposals with the RT-DETRv2 decoder. |

The wheat setting uses **500 queries**, anchor scale **`g = 0.04`**, and area penalty **`lambda = 8`**:

```text
Image → Backbone → Context–detail exchange → Encoder proposals
                                                ↓
                                 rank = objectness − λ × (w × h)
                                                ↓
                                     Top-K → Decoder → Detections
```

<a id="results"></a>
## Results

### Detection accuracy and speed

Protocol A · GWHD2021 · `1024 × 1024` · AP100 · A40 FP32 throughput, batch size 1.

| Model | AP100 ↑ | AP50 ↑ | AP75 ↑ | AP-S ↑ | FPS ↑ |
|---|---:|---:|---:|---:|---:|
| YOLO11m | 50.71 | 91.58 | 56.84 | 47.91 | **145.2** |
| RT-DETRv2-R50 | 50.68 | 91.74 | 56.92 | 47.96 | 108.5 |
| **TSECAF-DETR-R50** | **53.62** | **93.18** | **60.41** | **51.46** | 102.8 |

<p align="center">
  <img src="assets/readme/accuracy_efficiency.png" alt="Detection accuracy versus inference throughput" width="760">
</p>

### Using the decoder budget

Protocol A · common AP300 evaluation · each query budget trained independently.

| Configuration | Queries | AP300 ↑ | Latency ↓ |
|---|---:|---:|---:|
| RT-DETRv2-R50 | 700 | 50.75 | 10.12 ms |
| **TSECAF-DETR-R50** | **300** | **52.91** | **9.24 ms** |

**2.16 points higher AP300, 400 fewer queries, and 0.88 ms lower latency.**

### Held-out evaluation and transfer

| Evaluation | Metric | Baseline | TSECAF-DETR | Change |
|---|---|---:|---:|---:|
| GWHD2021 held-out test · Protocol B | AP100 ↑ | 49.82 | **52.45** | +2.63 |
| GWHD2021 held-out test · Protocol B | Counting MAE ↓ | 7.439 | **5.866** | −1.573 |
| Wheat Ears · direct transfer | AP300 ↑ | 39.62 | **43.81** | +4.19 |
| SKU-110K · independent training | AP100 ↑ | 43.28 | **44.92** | +1.64 |

Wheat Ears uses 48 images with 6240 annotations and protocol B checkpoints without target-domain fine-tuning. SKU-110K uses independent task training with the wheat allocation settings retained.

### Qualitative detections

<table>
  <tr>
    <td><img src="assets/readme/case_01.png" alt="Dense wheat head detections, example 1" width="440"></td>
    <td><img src="assets/readme/case_02.png" alt="Dense wheat head detections, example 2" width="440"></td>
  </tr>
  <tr>
    <td><img src="assets/readme/case_03.png" alt="Dense wheat head detections, example 3" width="440"></td>
    <td><img src="assets/readme/case_04.png" alt="Dense wheat head detections, example 4" width="440"></td>
  </tr>
</table>

<a id="quick-start"></a>
## Quick Start

### 1. Install

```bash
git clone https://github.com/Nopon-Knowledge/TSECAF-DETR.git
cd TSECAF-DETR/tsecaf_detr
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install `torch` and `torchvision` wheels that match your CUDA runtime. The paper experiments use a single NVIDIA A40 GPU.

<details>
<summary><b>CUDA runtime option</b></summary>

Some cluster environments need cuDNN disabled for stable RT-DETRv2 training:

```bash
export RTDETR_DISABLE_CUDNN=1
```

</details>

### 2. Prepare data

Place COCO-style datasets under `tsecaf_detr/datasets`, or point to an existing dataset directory:

```bash
export WHEAT_DATASETS_ROOT=/path/to/datasets
```

See [dataset protocols and directory layout](#datasets) below.

### 3. Train and evaluate

Run all following commands from `tsecaf_detr/`.

```bash
# Train TSECAF-DETR-R50
python tools/train_gwhd_a40.py --preset tsecaf_detr_r50

# Evaluate on GWHD2021
python tools/test_gwhd_a40.py --preset tsecaf_detr_r50

# Visualize predictions
python tools/visualize_gwhd_predictions.py --preset tsecaf_detr_r50_val_best
```

Training outputs are saved under `output/`. Datasets and pretrained checkpoints are downloaded separately.

<details>
<summary><b>Model scales and paired baselines</b></summary>

| Backbone | TSECAF-DETR preset | Baseline preset |
|---|---|---|
| R18 | `tsecaf_detr_r18` | `baseline_r18` |
| R34 | `tsecaf_detr_r34` | `baseline_r34` |
| R50 | `tsecaf_detr_r50` | `baseline_r50` |

```bash
python tools/train_gwhd_a40.py --preset baseline_r50
python tools/test_gwhd_a40.py --preset baseline_r50
```

Legacy preset and configuration aliases remain available for older experiment logs.

</details>

<details>
<summary><b>Direct Wheat Ears evaluation</b></summary>

Use the corresponding GWHD2021-trained checkpoint:

```bash
python tools/test_gwhd_a40.py --preset wheat_ears_tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset wheat_ears_baseline_r50
```

For the current manuscript, use protocol B checkpoints and the 48-image evaluation subset. AP300 denotes `maxDets=[1,10,300]`; AP100 denotes `maxDets=[1,10,100]`.

</details>

<details>
<summary><b>Profile parameters and computation</b></summary>

```bash
python tools/run_profile.py \
  -c configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml
```

</details>

<a id="datasets"></a>
## Datasets

| Protocol / evaluation | Partition | Role |
|---|---|---|
| **A · GWHD2021** | Random: 5499 train / 1016 validation | Detector comparisons and mechanism analysis |
| **B · GWHD2021** | Random: 4399 train / 1100 development / 1016 test | Held-out detection and counting |
| **Wheat Ears** | Random evaluation subset: 48 images / 6240 annotations | Direct transfer from protocol B |
| **C · SKU-110K** | Official dataset partition | Independently trained dense retail detection |

<details>
<summary><b>Expected COCO directory layout</b></summary>

```text
datasets/
├── gwhd_2021/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
└── Wheat_Ears_Detection_Dataset_rtdetr/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

Dataset configuration files control image and annotation paths. Point them to the intended partition when using a different protocol.

</details>

<details>
<summary><b>Conversion utilities and additional project evaluations</b></summary>

```bash
python tools/prepare_wheat_datasets.py --dataset wheat_ears
python tools/prepare_wheat_datasets.py --dataset global_wheat_codalab
```

Global Wheat CodaLab conversion and evaluation presets remain available for project experiments. These results are outside the current manuscript.

```bash
python tools/test_gwhd_a40.py --preset global_wheat_codalab_tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_baseline_r50
```

</details>

<a id="paper-to-code"></a>
## Paper to Code

| Component | Implementation |
|---|---|
| Detail preparation and statistics-guided context–detail exchange | [hybrid_encoder.py](tsecaf_detr/src/zoo/rtdetr/hybrid_encoder.py) |
| Anchor geometry and area-penalized query selection | [rtdetrv2_decoder.py](tsecaf_detr/src/zoo/rtdetr/rtdetrv2_decoder.py) |
| Full R50 configuration | [tsecaf_detr_r50vd_180e_gwhd.yml](tsecaf_detr/configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml) |
| Baselines, model scales, and ablations | [configs/rtdetrv2](tsecaf_detr/configs/rtdetrv2/) |
| Training presets | [train_gwhd_a40.py](tsecaf_detr/tools/train_gwhd_a40.py) |
| Evaluation presets | [test_gwhd_a40.py](tsecaf_detr/tools/test_gwhd_a40.py) |
| Dataset conversion | [prepare_wheat_datasets.py](tsecaf_detr/tools/prepare_wheat_datasets.py) |
| Additional tool usage | [tools/README.md](tsecaf_detr/tools/README.md) |

The configuration key `area_penalty` corresponds to the paper's coefficient λ. `anchor_grid_size` corresponds to g. The legacy `query_size_prior` key remains supported.

<details>
<summary><b>Repository structure and release notes</b></summary>

```text
TSECAF-DETR/
├── assets/readme/             # Architecture, results, and qualitative figures
├── tsecaf_detr/
│   ├── configs/
│   │   ├── dataset/           # COCO-style dataset configurations
│   │   └── rtdetrv2/          # Models, baselines, ablations, and transfer
│   ├── src/zoo/rtdetr/        # Detector implementation
│   └── tools/                # Training, evaluation, visualization, and export
└── LICENSE
```

External comparison detectors such as YOLO, D-FINE, and LW-DETR are not vendored in this release. Generated checkpoints, logs, exported models, and visualization outputs are ignored by Git. Exact table reproduction requires the corresponding partition, trained checkpoint, and per-seed evaluation outputs.

</details>

## License and Acknowledgments

Released under the [repository license](LICENSE). This implementation builds on **RT-DETRv2** and uses public detection datasets. Please cite the associated TSECAF-DETR paper after publication, together with RT-DETRv2 and the datasets used in your experiments.

---

<p align="center">
  <a href="#top">Back to top ↑</a>
</p>
