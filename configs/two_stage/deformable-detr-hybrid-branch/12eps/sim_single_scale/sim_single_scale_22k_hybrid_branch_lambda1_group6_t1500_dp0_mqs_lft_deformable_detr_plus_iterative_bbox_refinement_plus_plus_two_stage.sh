#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-hybrid-branch/12eps/swin/sim_single_scale_22k_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --epochs 12 \
    --use_wandb \
    --lr_drop 11 \
    --num_features_levels 1 \
    --num_queries_one2one 300 \
    --num_queries_one2many 1500 \
    --lr_backbone_names body.backbone \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --backbone sim_single_scale \
    --pretrained_backbone_path /data/pwojcik/SimMIM/TCGA_256/checkpoint-latest.pth \
    ${PY_ARGS}
