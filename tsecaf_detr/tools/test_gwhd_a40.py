"""Evaluation launcher for TSECAF-DETR wheat-head experiments on a single A40."""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
try:
    from project_paths import PROJECT_ROOT
except ModuleNotFoundError:
    from tools.project_paths import PROJECT_ROOT


def _checkpoint(run_name: str) -> str:
    return str(PROJECT_ROOT / "output" / run_name / "best.pth")


def _eval_dir(*parts: str) -> str:
    return str(PROJECT_ROOT / "output" / "paper_eval" / "_".join(parts))


EVAL_PRESETS = {
    "baseline_r18": {
        "description": "Evaluate the RT-DETRv2-R18 GWHD2021 baseline.",
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_gwhd_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r18vd_120e_gwhd_baseline"),
        "output_dir": _eval_dir("gwhd", "r18", "baseline"),
    },
    "baseline_r34": {
        "description": "Evaluate the RT-DETRv2-R34 GWHD2021 baseline.",
        "config": "configs/rtdetrv2/rtdetrv2_r34vd_180e_gwhd_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r34vd_180e_gwhd_baseline"),
        "output_dir": _eval_dir("gwhd", "r34", "baseline"),
    },
    "baseline_r50": {
        "description": "Evaluate the RT-DETRv2-R50 GWHD2021 baseline.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r50vd_180e_gwhd_baseline"),
        "output_dir": _eval_dir("gwhd", "r50", "baseline"),
    },
    "small_object_aware_rtdetr_r18": {
        "description": "Legacy alias for evaluating TSECAF-DETR-R18 on GWHD2021.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r18vd_120e_gwhd.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r18vd_120e_gwhd"),
        "output_dir": _eval_dir("gwhd", "r18", "small_object_aware"),
    },
    "small_object_aware_rtdetr_r34": {
        "description": "Legacy alias for evaluating TSECAF-DETR-R34 on GWHD2021.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r34vd_180e_gwhd.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r34vd_180e_gwhd"),
        "output_dir": _eval_dir("gwhd", "r34", "small_object_aware"),
    },
    "small_object_aware_rtdetr_r50": {
        "description": "Legacy alias for evaluating TSECAF-DETR-R50 on GWHD2021.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r50vd_180e_gwhd"),
        "output_dir": _eval_dir("gwhd", "r50", "small_object_aware"),
    },
    "wheat_ears_baseline_r18": {
        "description": "Direct transfer: R18 baseline trained on GWHD2021, evaluated on Wheat Ears.",
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_eval_wheat_ears_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r18vd_120e_gwhd_baseline"),
        "output_dir": _eval_dir("wheat_ears", "r18", "baseline"),
    },
    "wheat_ears_small_object_aware_r18": {
        "description": "Legacy alias for TSECAF-DETR-R18 direct transfer to Wheat Ears.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r18vd_eval_wheat_ears.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r18vd_120e_gwhd"),
        "output_dir": _eval_dir("wheat_ears", "r18", "small_object_aware"),
    },
    "wheat_ears_baseline_r34": {
        "description": "Direct transfer: R34 baseline trained on GWHD2021, evaluated on Wheat Ears.",
        "config": "configs/rtdetrv2/rtdetrv2_r34vd_eval_wheat_ears_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r34vd_180e_gwhd_baseline"),
        "output_dir": _eval_dir("wheat_ears", "r34", "baseline"),
    },
    "wheat_ears_small_object_aware_r34": {
        "description": "Legacy alias for TSECAF-DETR-R34 direct transfer to Wheat Ears.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r34vd_eval_wheat_ears.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r34vd_180e_gwhd"),
        "output_dir": _eval_dir("wheat_ears", "r34", "small_object_aware"),
    },
    "wheat_ears_baseline_r50": {
        "description": "Direct transfer: R50 baseline trained on GWHD2021, evaluated on Wheat Ears.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_eval_wheat_ears_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r50vd_180e_gwhd_baseline"),
        "output_dir": _eval_dir("wheat_ears", "r50", "baseline"),
    },
    "wheat_ears_small_object_aware_r50": {
        "description": "Legacy alias for TSECAF-DETR-R50 direct transfer to Wheat Ears.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r50vd_eval_wheat_ears.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r50vd_180e_gwhd"),
        "output_dir": _eval_dir("wheat_ears", "r50", "small_object_aware"),
    },
    "global_wheat_codalab_baseline_r18": {
        "description": "Direct transfer: R18 baseline trained on GWHD2021, evaluated on Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_eval_global_wheat_codalab_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r18vd_120e_gwhd_baseline"),
        "output_dir": _eval_dir("global_wheat_codalab", "r18", "baseline"),
    },
    "global_wheat_codalab_small_object_aware_r18": {
        "description": "Legacy alias for TSECAF-DETR-R18 direct transfer to Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r18vd_eval_global_wheat_codalab.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r18vd_120e_gwhd"),
        "output_dir": _eval_dir("global_wheat_codalab", "r18", "small_object_aware"),
    },
    "global_wheat_codalab_baseline_r34": {
        "description": "Direct transfer: R34 baseline trained on GWHD2021, evaluated on Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/rtdetrv2_r34vd_eval_global_wheat_codalab_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r34vd_180e_gwhd_baseline"),
        "output_dir": _eval_dir("global_wheat_codalab", "r34", "baseline"),
    },
    "global_wheat_codalab_small_object_aware_r34": {
        "description": "Legacy alias for TSECAF-DETR-R34 direct transfer to Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r34vd_eval_global_wheat_codalab.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r34vd_180e_gwhd"),
        "output_dir": _eval_dir("global_wheat_codalab", "r34", "small_object_aware"),
    },
    "global_wheat_codalab_baseline_r50": {
        "description": "Direct transfer: R50 baseline trained on GWHD2021, evaluated on Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_eval_global_wheat_codalab_baseline.yml",
        "resume": _checkpoint("rtdetrv2_r50vd_180e_gwhd_baseline"),
        "output_dir": _eval_dir("global_wheat_codalab", "r50", "baseline"),
    },
    "global_wheat_codalab_small_object_aware_r50": {
        "description": "Legacy alias for TSECAF-DETR-R50 direct transfer to Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r50vd_eval_global_wheat_codalab.yml",
        "resume": _checkpoint("small_object_aware_rtdetr_r50vd_180e_gwhd"),
        "output_dir": _eval_dir("global_wheat_codalab", "r50", "small_object_aware"),
    },
}

EVAL_PRESETS.update({
    "tsecaf_detr_r18": {
        **EVAL_PRESETS["small_object_aware_rtdetr_r18"],
        "description": "Evaluate TSECAF-DETR-R18 on GWHD2021.",
        "config": "configs/rtdetrv2/tsecaf_detr_r18vd_120e_gwhd.yml",
        "resume": _checkpoint("tsecaf_detr_r18vd_120e_gwhd"),
        "output_dir": _eval_dir("gwhd", "r18", "tsecaf_detr"),
    },
    "tsecaf_detr_r34": {
        **EVAL_PRESETS["small_object_aware_rtdetr_r34"],
        "description": "Evaluate TSECAF-DETR-R34 on GWHD2021.",
        "config": "configs/rtdetrv2/tsecaf_detr_r34vd_180e_gwhd.yml",
        "resume": _checkpoint("tsecaf_detr_r34vd_180e_gwhd"),
        "output_dir": _eval_dir("gwhd", "r34", "tsecaf_detr"),
    },
    "tsecaf_detr_r50": {
        **EVAL_PRESETS["small_object_aware_rtdetr_r50"],
        "description": "Evaluate TSECAF-DETR-R50 on GWHD2021.",
        "config": "configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml",
        "resume": _checkpoint("tsecaf_detr_r50vd_180e_gwhd"),
        "output_dir": _eval_dir("gwhd", "r50", "tsecaf_detr"),
    },
    "wheat_ears_tsecaf_detr_r18": {
        **EVAL_PRESETS["wheat_ears_small_object_aware_r18"],
        "description": "Direct transfer: TSECAF-DETR-R18 trained on GWHD2021, evaluated on Wheat Ears.",
        "config": "configs/rtdetrv2/tsecaf_detr_r18vd_eval_wheat_ears.yml",
        "resume": _checkpoint("tsecaf_detr_r18vd_120e_gwhd"),
        "output_dir": _eval_dir("wheat_ears", "r18", "tsecaf_detr"),
    },
    "wheat_ears_tsecaf_detr_r34": {
        **EVAL_PRESETS["wheat_ears_small_object_aware_r34"],
        "description": "Direct transfer: TSECAF-DETR-R34 trained on GWHD2021, evaluated on Wheat Ears.",
        "config": "configs/rtdetrv2/tsecaf_detr_r34vd_eval_wheat_ears.yml",
        "resume": _checkpoint("tsecaf_detr_r34vd_180e_gwhd"),
        "output_dir": _eval_dir("wheat_ears", "r34", "tsecaf_detr"),
    },
    "wheat_ears_tsecaf_detr_r50": {
        **EVAL_PRESETS["wheat_ears_small_object_aware_r50"],
        "description": "Direct transfer: TSECAF-DETR-R50 trained on GWHD2021, evaluated on Wheat Ears.",
        "config": "configs/rtdetrv2/tsecaf_detr_r50vd_eval_wheat_ears.yml",
        "resume": _checkpoint("tsecaf_detr_r50vd_180e_gwhd"),
        "output_dir": _eval_dir("wheat_ears", "r50", "tsecaf_detr"),
    },
    "global_wheat_codalab_tsecaf_detr_r18": {
        **EVAL_PRESETS["global_wheat_codalab_small_object_aware_r18"],
        "description": "Direct transfer: TSECAF-DETR-R18 trained on GWHD2021, evaluated on Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/tsecaf_detr_r18vd_eval_global_wheat_codalab.yml",
        "resume": _checkpoint("tsecaf_detr_r18vd_120e_gwhd"),
        "output_dir": _eval_dir("global_wheat_codalab", "r18", "tsecaf_detr"),
    },
    "global_wheat_codalab_tsecaf_detr_r34": {
        **EVAL_PRESETS["global_wheat_codalab_small_object_aware_r34"],
        "description": "Direct transfer: TSECAF-DETR-R34 trained on GWHD2021, evaluated on Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/tsecaf_detr_r34vd_eval_global_wheat_codalab.yml",
        "resume": _checkpoint("tsecaf_detr_r34vd_180e_gwhd"),
        "output_dir": _eval_dir("global_wheat_codalab", "r34", "tsecaf_detr"),
    },
    "global_wheat_codalab_tsecaf_detr_r50": {
        **EVAL_PRESETS["global_wheat_codalab_small_object_aware_r50"],
        "description": "Direct transfer: TSECAF-DETR-R50 trained on GWHD2021, evaluated on Global Wheat CodaLab.",
        "config": "configs/rtdetrv2/tsecaf_detr_r50vd_eval_global_wheat_codalab.yml",
        "resume": _checkpoint("tsecaf_detr_r50vd_180e_gwhd"),
        "output_dir": _eval_dir("global_wheat_codalab", "r50", "tsecaf_detr"),
    },
})


def configure_torch_runtime() -> None:
    # Match the training launcher: some cluster CUDA setups expose GPUs but
    # fail cuDNN engine selection on the first conv unless cuDNN is disabled.
    if os.environ.get("RTDETR_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None}
    )

    cfg = YAMLConfig(args.config, **update_dict)
    print("cfg: ", cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.val()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RT-DETRv2 and TSECAF-DETR wheat-head checkpoints with preset defaults."
    )

    parser.add_argument("--preset", choices=sorted(EVAL_PRESETS), default="tsecaf_detr_r18")
    parser.add_argument("-c", "--config", type=str, help="Config path. Defaults to the selected preset.")
    parser.add_argument("-r", "--resume", type=str, help="checkpoint for evaluation")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summary")

    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    preset = EVAL_PRESETS[args.preset]
    if args.config is None:
        args.config = preset["config"]
    if args.resume is None:
        args.resume = preset["resume"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]

    print(f"Selected preset: {args.preset} - {preset['description']}")
    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {args.resume}")
    print(f"Evaluation outputs: {args.output_dir}")

    wants_cuda = args.device is None or str(args.device).lower() != "cpu"
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is visible in the current session. "
            "This A40 evaluator should be run inside a GPU allocation/job. "
            "If you intentionally want CPU evaluation, pass '-d cpu'."
        )

    configure_torch_runtime()
    main(args)
