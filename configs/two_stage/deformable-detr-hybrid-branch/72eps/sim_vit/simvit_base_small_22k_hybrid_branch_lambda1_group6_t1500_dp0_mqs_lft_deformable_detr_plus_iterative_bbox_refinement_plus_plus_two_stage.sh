#!/usr/bin/env bash

set -x

EXP_DIR=exps/two_stage/deformable-detr-hybrid-branch/72eps/swin/simvit_base_22k_hybrid_branch_lambda1_group6_t1500_dp0_mqs_lft_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --lr_backbone_names body.backbone \
    --dim_feedforward 2048 \
    --epochs 72 \
    --lr_drop 66 \
    --num_queries_one2one 300 \
    --num_queries_one2many 1500 \
    --k_one2many 3 \
    --lambda_one2many 1.0 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --backbone simvit_base \
    --pretrained_backbone_path /data/pwojcik/SimMIM/TCGA_256/checkpoint-latest.pth \
    ${PY_ARGS}
