"""Training launcher for TSECAF-DETR wheat-head experiments on a single A40."""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
try:
    from project_paths import REPO_ROOT, resolve_first_existing
except ModuleNotFoundError:
    from tools.project_paths import REPO_ROOT, resolve_first_existing


def checkpoint_candidates(*filenames: str) -> list[Path]:
    pretrained_dir = os.environ.get("RTDETR_PRETRAINED_DIR")
    candidates: list[Path] = []
    for filename in filenames:
        if pretrained_dir:
            candidates.append(Path(pretrained_dir).expanduser() / filename)
        candidates.extend([
            REPO_ROOT / filename,
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename,
        ])
    return candidates


TRAINING_PRESETS = {
    "baseline_r18": {
        "description": "RT-DETRv2-R18 GWHD2021 baseline used in the paper.",
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_gwhd_baseline.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    },
    "baseline_r34": {
        "description": "RT-DETRv2-R34 GWHD2021 baseline used in the paper.",
        "config": "configs/rtdetrv2/rtdetrv2_r34vd_180e_gwhd_baseline.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r34vd_120e_coco_ema.pth"),
    },
    "baseline_r50": {
        "description": "RT-DETRv2-R50 GWHD2021 baseline used in the paper.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_baseline.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r50vd_6x_coco_ema.pth"),
    },
    "small_object_aware_rtdetr_r18": {
        "description": "Legacy alias for TSECAF-DETR-R18.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r18vd_120e_gwhd.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    },
    "small_object_aware_rtdetr_r34": {
        "description": "Legacy alias for TSECAF-DETR-R34.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r34vd_180e_gwhd.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r34vd_120e_coco_ema.pth"),
    },
    "small_object_aware_rtdetr_r50": {
        "description": "Legacy alias for TSECAF-DETR-R50.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r50vd_6x_coco_ema.pth"),
    },
    "ablation_baseline_r50": {
        "description": "RT-DETRv2-R50 controlled reference for pathway ablation.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_baseline.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r50vd_6x_coco_ema.pth"),
    },
    "ablation_wheat_fusion_r50": {
        "description": "Pathway probe: context-detail representation before proposal generation.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_ablation_wheat_fusion.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r50vd_6x_coco_ema.pth"),
    },
    "ablation_detail_enhance_r50": {
        "description": "Pathway probe: detail-branch emphasis at the high-resolution evidence stage.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_ablation_detail_enhance.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r50vd_6x_coco_ema.pth"),
    },
    "ablation_agnostic_small_r50": {
        "description": "Pathway probe: compact-query allocation before decoder refinement.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_ablation_agnostic_small.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r50vd_6x_coco_ema.pth"),
    },
    "ablation_full_small_object_aware_r50": {
        "description": "Full TSECAF-DETR-R50 evidence-to-query pathway.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml",
        "tuning_candidates": checkpoint_candidates("rtdetrv2_r50vd_6x_coco_ema.pth"),
    },
}

TRAINING_PRESETS.update({
    "tsecaf_detr_r18": {
        **TRAINING_PRESETS["small_object_aware_rtdetr_r18"],
        "description": "TSECAF-DETR-R18 representation-query co-design on GWHD2021.",
        "config": "configs/rtdetrv2/tsecaf_detr_r18vd_120e_gwhd.yml",
    },
    "tsecaf_detr_r34": {
        **TRAINING_PRESETS["small_object_aware_rtdetr_r34"],
        "description": "TSECAF-DETR-R34 representation-query co-design on GWHD2021.",
        "config": "configs/rtdetrv2/tsecaf_detr_r34vd_180e_gwhd.yml",
    },
    "tsecaf_detr_r50": {
        **TRAINING_PRESETS["small_object_aware_rtdetr_r50"],
        "description": "TSECAF-DETR-R50 representation-query co-design on GWHD2021.",
        "config": "configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml",
    },
    "pathway_context_detail_r50": {
        **TRAINING_PRESETS["ablation_wheat_fusion_r50"],
        "description": "Pathway probe: context-detail representation before proposal generation.",
    },
    "pathway_detail_branch_r50": {
        **TRAINING_PRESETS["ablation_detail_enhance_r50"],
        "description": "Pathway probe: detail-branch emphasis at the high-resolution evidence stage.",
    },
    "pathway_compact_query_r50": {
        **TRAINING_PRESETS["ablation_agnostic_small_r50"],
        "description": "Pathway probe: compact-query allocation before decoder refinement.",
    },
    "pathway_full_tsecaf_r50": {
        **TRAINING_PRESETS["ablation_full_small_object_aware_r50"],
        "description": "Full TSECAF-DETR-R50 evidence-to-query pathway.",
    },
})


def configure_torch_runtime() -> None:
    # Some cluster CUDA environments expose GPUs but fail on cuDNN engine
    # selection for the first backbone conv. Default to the more conservative
    # native CUDA kernels for this custom training launcher.
    if os.environ.get("RTDETR_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def resolve_checkpoint(candidates) -> Optional[str]:
    return resolve_first_existing(*candidates)


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), (
        "Only support from_scrach or resume or tuning at one time"
    )

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None}
    )

    cfg = YAMLConfig(args.config, **update_dict)
    print("cfg: ", cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RT-DETRv2 and TSECAF-DETR wheat-head detectors with preset hyperparameters."
    )

    parser.add_argument("--preset", choices=sorted(TRAINING_PRESETS), default="tsecaf_detr_r18")
    parser.add_argument("-c", "--config", type=str, help="Config path. Defaults to the selected preset.")
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summary")
    parser.add_argument("--test-only", action="store_true", default=False)

    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    preset = TRAINING_PRESETS[args.preset]
    if args.config is None:
        args.config = preset["config"]

    if args.resume is None and args.tuning is None:
        args.tuning = resolve_checkpoint(preset["tuning_candidates"])

    print(f"Selected preset: {args.preset} - {preset['description']}")
    print(f"Using config: {args.config}")
    if args.tuning:
        print(f"Using tuning checkpoint: {args.tuning}")
    else:
        print("No local tuning checkpoint found, training will start without external pretrained weights.")

    wants_cuda = args.device is None or str(args.device).lower() != "cpu"
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible in the current session. "
            "This A40 launcher should be run inside a GPU allocation/job. "
            "If you intentionally want CPU training, pass '-d cpu'."
        )

    configure_torch_runtime()
    main(args)
