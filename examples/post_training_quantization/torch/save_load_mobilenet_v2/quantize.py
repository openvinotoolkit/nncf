# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Tuple

import torch

import nncf
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import QUANTIZED_CHECKPOINT_FILE_NAME
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import ROOT
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import get_data_loader
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import get_device
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import get_mobilenet_v2

val_data_loader = get_data_loader()
batch_size = val_data_loader.batch_size

device = get_device()
torch_model = get_mobilenet_v2(device)
torch_model.eval()

###############################################################################
# Quantize a PyTorch model

# The transformation function transforms a data item into model input data.
#
# To validate the transform function use the following code:
# >> for data_item in val_loader:
# >>    model(transform_fn(data_item, device))


def transform_fn(data_item: Tuple[torch.Tensor, int], device: torch.device) -> torch.Tensor:
    images, _ = data_item
    return images.to(device)


# The calibration dataset is a small, no label, representative dataset
# (~100-500 samples) that is used to estimate the range, i.e. (min, max) of all
# floating point activation tensors in the model, to initialize the quantization
# parameters.

# The easiest way to define a calibration dataset is to use a training or
# validation dataset and a transformation function to remove labels from the data
# item and prepare model input data. The quantize method uses a small subset
# (default: 300 samples) of the calibration dataset.

# Recalculation default subset_size parameter based on batch_size.

print("[1/2] Quantize the original PyTorch model")
subset_size = 300 // batch_size
calibration_dataset = nncf.Dataset(val_data_loader, partial(transform_fn, device=device))
torch_quantized_model = nncf.quantize(torch_model, calibration_dataset, subset_size=subset_size)

print("[2/2] Save quantized model to a checkpoint")
state_dict = torch_quantized_model.state_dict()
nncf_config = torch_quantized_model.nncf.get_config()

quantized_checkpoint_path = ROOT / QUANTIZED_CHECKPOINT_FILE_NAME
torch.save({"model_state_dict": state_dict, "nncf_config": nncf_config}, quantized_checkpoint_path)
print(f"Quantized model saved to {quantized_checkpoint_path}")
