# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy
import torchvision.transforms as T

import wandb
import torch
import numpy as np
from tqdm import tqdm
import errno
from skimage import io
from torchvision.utils import save_image

import util.misc as utils
from datasets.coco_eval import CocoEvaluator, convert_to_xywh
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util import box_ops

scaler = torch.cuda.amp.GradScaler()

def clip_bbox(bbox_tensor):
    # Clip bounding box coordinates to the interval [0, 255]
    bbox_tensor[:, 0].clamp_(min=0, max=255)  # xmin
    bbox_tensor[:, 1].clamp_(min=0, max=255)  # ymin
    bbox_tensor[:, 2].clamp_(min=0, max=255)  # xmax
    bbox_tensor[:, 3].clamp_(min=0, max=255)  # ymax
    return bbox_tensor


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many):
    # one-to-one-loss
    loss_dict = criterion(outputs, targets)
    multi_targets = copy.deepcopy(targets)
    # repeat the targets
    for target in multi_targets:
        target["boxes"] = target["boxes"].repeat(k_one2many, 1)
        target["labels"] = target["labels"].repeat(k_one2many)

    outputs_one2many = dict()
    outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
    outputs_one2many["pred_boxes"] = outputs["pred_boxes_one2many"]
    outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]

    # one-to-many loss
    loss_dict_one2many = criterion(outputs_one2many, multi_targets)
    for key, value in loss_dict_one2many.items():
        if key + "_one2many" in loss_dict.keys():
            loss_dict[key + "_one2many"] += value * lambda_one2many
        else:
            loss_dict[key + "_one2many"] = value * lambda_one2many
    return loss_dict


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    k_one2many=1,
    lambda_one2many=1.0,
    use_wandb=False,
    use_fp16=False,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    metric_logger.add_meter(
        "grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        with torch.cuda.amp.autocast() if use_fp16 else torch.cuda.amp.autocast(
            enabled=False
        ):
            if use_fp16:
                optimizer.zero_grad()
            outputs = model(samples)

            if k_one2many > 0:
                loss_dict = train_hybrid(
                    outputs, targets, k_one2many, criterion, lambda_one2many
                )
            else:
                loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if use_fp16:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
        else:
            optimizer.zero_grad()
            losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm
            )
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

        if use_wandb:
            try:
                wandb.log(loss_dict)
            except:
                pass
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def predict_prompts(prompts_paths, dataset_name, model, postprocessors):
    mkdir(f'/data/pwojcik/PromptNucSeg/segmentor/prompts/pannuke123_boxes/{prompts_paths}')
    print('Test files')
    test_files = np.load(f'/data/pwojcik/PromptNucSeg/segmentor/datasets/{dataset_name}_test_files.npy')
    process_files(prompts_paths, test_files, model, postprocessors)
    print('Val files')
    val_files = np.load(f'/data/pwojcik/PromptNucSeg/segmentor/datasets/{dataset_name}_val_files.npy')
    process_files(prompts_paths, val_files, model, postprocessors)

SCORE_THRESHOLD = 0.33


def process_files(prompts_paths, files, model, postprocessors):
    for file in sorted(tqdm(files)):
        img = io.imread(f'/data/pwojcik/PromptNucSeg/segmentor/{file}')[..., :3]

        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        image = normalize(img)
        samples = utils.nested_tensor_from_tensor_list([image]).to('cuda')
        outputs = model(samples)

        orig_target_sizes = torch.stack([torch.as_tensor([256, 256])], dim=0).to('cuda')
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        scores = results[0]['scores']
        boxes = results[0]['boxes']
        labels = results[0]['labels']

        boxes = boxes[scores >= SCORE_THRESHOLD]
        labels = labels[scores >= SCORE_THRESHOLD]
        labels = labels - 1

        boxes = clip_bbox(boxes)

        boxes = boxes.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        save_content = np.concatenate([boxes, labels[:, None]], axis=-1)
        #print(save_content)
        np.save(
            f'/data/pwojcik/PromptNucSeg/segmentor/prompts/pannuke123_boxes/{file.split("/")[-1][:-4]}',
            save_content
        )


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    use_wandb=False,
):
    # disable the one-to-many branch queries
    # save them frist
    save_num_queries = model.module.num_queries
    save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
    model.module.num_queries = model.module.num_queries_one2one
    model.module.transformer.two_stage_num_proposals = model.module.num_queries

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    results_all = {}
    target_all = {}

    #print('Starting to produce prompts')
    #predict_prompts(prompts_paths='prompts_boxes', dataset_name='pannuke123',
    #                model=model, postprocessors=postprocessors)
    #print('Done')

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        #print(len(samples))
        #print(samples.mask.shape)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        img = torch.tensor(samples.tensors)
        bs = img.shape[0]
        for i in range(bs):
            #print(targets[i].keys())
            image_id = targets[i]["image_id"].item()
            #print(targets[i]["labels"])
            im = img[i]
            #print('!!!')
            #print(im.shape)
            scores = results[i]['scores'] >= 0.355
            boxes = results[i]['boxes'].clone()
            boxes_r = targets[i]['boxes'].clone()
            boxes_r = box_ops.box_cxcywh_to_xyxy(boxes_r) * 800
            #print(boxes_r)
            results_all.update({image_id: (results[i]['scores'], results[i]['boxes'], results[i]['labels'])})
            target_all.update({image_id: (targets[i]['boxes'], targets[i]['labels'])})

            from torchvision.utils import draw_bounding_boxes
            im = (im * 255).clamp(0, 255).to(torch.uint8)
            #print(boxes)
            boxes = boxes * (800 / 256)
            drawn_boxes = draw_bounding_boxes(im, boxes[scores], colors="red")
            drawn_boxes = draw_bounding_boxes(drawn_boxes, boxes_r, colors="blue")
            import torchvision.transforms.functional as TF
            image = TF.to_pil_image(drawn_boxes)
            #image.save(f'/data/pwojcik/detr_dump/img_{image_id}.png')
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
    import pickle
    #print('!!!')
    #print(len(results_all.keys()))

    #with open('/data/pwojcik/detr_dump4/results.pkl', 'wb') as f:
    #    pickle.dump(results_all, f)
    #with open('/data/pwojcik/detr_dump4/target.pkl', 'wb') as f:
    #    pickle.dump(target_all, f)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]
    if use_wandb:
        try:
            wandb.log({"AP": stats["coco_eval_bbox"][0]})
            wandb.log(stats)
        except:
            pass

    # recover the model parameters for next training epoch
    model.module.num_queries = save_num_queries
    model.module.transformer.two_stage_num_proposals = save_two_stage_num_proposals
    return stats, coco_evaluator
