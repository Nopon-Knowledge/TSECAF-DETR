"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

from typing import Iterable, List, Optional

from ...core import register
from faster_coco_eval.utils.pytorch import FasterCocoEvaluator


@register()
class CocoEvaluator(FasterCocoEvaluator):
    def __init__(self, coco_gt, iou_types, lvis_style: bool = False, max_dets=None):
        super().__init__(coco_gt=coco_gt, iou_types=iou_types, lvis_style=lvis_style)
        self.max_dets = self._normalize_max_dets(max_dets)
        self._apply_max_dets()

    @staticmethod
    def _normalize_max_dets(max_dets) -> Optional[List[int]]:
        if max_dets is None:
            return None
        if isinstance(max_dets, int):
            return [1, 10, max_dets]
        if isinstance(max_dets, Iterable):
            values = [int(value) for value in max_dets]
            if len(values) != 3:
                raise ValueError(f"max_dets must contain exactly 3 values, got {values}")
            return values
        raise TypeError(f"Unsupported max_dets type: {type(max_dets)}")

    def _apply_max_dets(self) -> None:
        if self.max_dets is None:
            return
        for evaluator in self.coco_eval.values():
            evaluator.params.maxDets = list(self.max_dets)

    def cleanup(self):
        super().cleanup()
        self._apply_max_dets()
