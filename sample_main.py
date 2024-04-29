from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from PIL import Image

from datasets.coco import make_coco_transforms, CocoDetection
from util.misc import get_local_rank, get_local_size
from torchvision.utils import save_image
import datasets.transforms as T


if __name__ == "__main__":

    args = {"coco_path": "/Users/piotrwojcik/Downloads/dataset/", "masks": False, "cache_mode": False}

    def build_data(image_set, args, eval_in_training_set):
        root = Path(args['coco_path'])
        assert root.exists(), f"provided COCO path {root} does not exist"
        mode = "instances"
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
            "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
            "test": (root / "test2017", root / "annotations" / f"{mode}_test2017.json")
        }

        img_folder, ann_file = PATHS[image_set]
        if eval_in_training_set:
            image_set = "val"
            print("use validation dataset transforms")
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(image_set),
            return_masks=args['masks'],
            cache_mode=args['cache_mode'],
            local_rank=get_local_rank(),
            local_size=get_local_size(),
        )
        return dataset

    data = build_data('train', args, eval_in_training_set=False)

    for idx, sample in enumerate(data):
        print(sample[0].shape)
        save_image(sample[0], f'/Users/piotrwojcik/Downloads/baseline/panuke_coco/img_{idx}.png')
