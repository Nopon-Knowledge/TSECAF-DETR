# Tools

Run these commands from `tsecaf_detr/`. Recommended presets use the final TSECAF-DETR naming; legacy aliases remain available for older experiment logs.

## Training

```bash
python tools/train_gwhd_a40.py --preset baseline_r50
python tools/train_gwhd_a40.py --preset tsecaf_detr_r50
```

## Evaluation

```bash
python tools/test_gwhd_a40.py --preset tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset wheat_ears_tsecaf_detr_r50
python tools/test_gwhd_a40.py --preset global_wheat_codalab_tsecaf_detr_r50
```

## Pathway Ablation

```bash
python tools/train_gwhd_a40.py --preset pathway_context_detail_r50
python tools/train_gwhd_a40.py --preset pathway_detail_branch_r50
python tools/train_gwhd_a40.py --preset pathway_compact_query_r50
python tools/train_gwhd_a40.py --preset pathway_full_tsecaf_r50
```

## Figures

```bash
python tools/visualize_gwhd_predictions.py --preset tsecaf_detr_r50_val_best
python tools/visualize_feature_maps.py --preset gwhd_r50_baseline_vs_tsecaf_detr
python tools/plot_training_curves_svg.py --preset gwhd_r50_baseline_vs_tsecaf_detr
python tools/plot_ablation_svg.py --preset gwhd_r50_tsecaf_pathway_ablation
python tools/plot_generalization_svg.py --preset r50_baseline_vs_tsecaf_detr_transfer
python tools/plot_pr_curves_svg.py --preset gwhd_r50_baseline_vs_tsecaf_detr
```

## Utilities

```bash
python tools/prepare_wheat_datasets.py --dataset wheat_ears
python tools/prepare_wheat_datasets.py --dataset global_wheat_codalab
python tools/run_profile.py -c configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml
python tools/export_onnx.py -c configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml -r output/tsecaf_detr_r50vd_180e_gwhd/best.pth -o tsecaf_detr_r50.onnx -s 1024 --check
```

Before reporting zero-shot transfer results, audit exact and near-duplicate overlap between the source training images and every target validation split:

```bash
python tools/audit_dataset_overlap.py \
  --train /path/to/gwhd/train/images \
  --eval /path/to/target/val/images \
  --threshold 4 \
  --output overlap_audit.json
```

The report combines byte-level SHA-256 matching with 64-bit dHash screening and lists every cross-set pair that requires manual review.
