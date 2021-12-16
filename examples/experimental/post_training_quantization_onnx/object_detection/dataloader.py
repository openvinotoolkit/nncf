import sys
import os
from typing import List
from typing import Union
import pathlib

import torch
from torch.utils.data import DataLoader, Dataset


# This class is necessary for adding the post-processing part from the original pipeline
class DatasetWithPostProcessing(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, labels_out, files, shapes = self.dataset[index]
        img = img.to(img, non_blocking=True).float() / 255
        return img, labels_out, files, shapes

    def __len__(self):
        return len(self.dataset)


def create_train_dataloader(dataset_dir: Union[str, bytes, os.PathLike], input_shape: List[int], stride: int,
                            batch_size: int = 1, num_workers: int = 4) -> torch.utils.data.DataLoader:
    image_size = [input_shape[-2], input_shape[-1]]
    try:
        # Include dataloader from yolov5 repo
        yolov5_rep_path = pathlib.Path().resolve() / 'yolov5'
        sys.path.append(str(yolov5_rep_path))
        sys.path.append(str(yolov5_rep_path / 'utils'))

        from datasets import InfiniteDataLoader
        from datasets import LoadImagesAndLabels

        original_dataset = LoadImagesAndLabels(dataset_dir + '/train2017.txt', image_size[0], batch_size)

        dataset_with_post_processing = DatasetWithPostProcessing(original_dataset)

        dataloader = InfiniteDataLoader(
            dataset_with_post_processing,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            sampler=None,
            pin_memory=True,
            collate_fn=original_dataset.collate_fn)
    except ImportError as e:
        sys.exit("Could not import dataloader from yolov5 repo")
    return dataloader
