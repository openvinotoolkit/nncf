# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from typing import Generator, Optional, TypeVar

import torch
from torch import nn

from nncf import Dataset
from nncf.common.logging.track_progress import track

TModel = TypeVar("TModel", bound=nn.Module)


@torch.no_grad()
def batch_norm_adaptation(
    model: TModel, *, calibration_dataset: Dataset, num_iterations: Optional[int] = None
) -> TModel:
    """
    Adapt the batch normalization layers of the given model using the provided dataset.

    This function runs a specified number of iterations (batches) through the model
    to update the running statistics of the batch normalization layers.

    :param model: The model to adapt.
    :param calibration_dataset: The dataset to use for the adaptation.
    :param num_iterations: The number of iterations (batches) to use for adaptation.
        If set to None, the adaptation will run for the entire dataset.
    """
    with set_batchnorm_train_only(model):
        total = calibration_dataset.get_length()
        if num_iterations is not None:
            total = min(num_iterations, total) if total is not None else num_iterations

        for idx, input_data in track(
            enumerate(calibration_dataset.get_inference_data()),
            total=total,
            description="Batch norm adaptation",
        ):
            if num_iterations is not None and idx >= num_iterations:
                break

            if isinstance(input_data, dict):
                model(**input_data)
            elif isinstance(input_data, tuple):
                model(*input_data)
            else:
                model(input_data)

    return model


@contextmanager
def set_batchnorm_train_only(model: nn.Module) -> Generator[None, None, None]:
    """
    Context manager that sets only BatchNorm modules to train mode,
    while keeping all other modules in eval mode.
    Restores the original training states afterward.

    :param model: The model.
    """
    # Store the original training states
    original_states = {}
    for name, module in model.named_modules():
        original_states[name] = module.training

    try:
        # Set all modules to eval, then only BN to train
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train()
        yield
    finally:
        # Restore original training states
        for name, module in model.named_modules():
            module.train(original_states[name])
