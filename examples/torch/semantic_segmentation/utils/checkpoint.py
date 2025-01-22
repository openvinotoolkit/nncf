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

import os

import torch

from examples.torch.common.model_loader import COMPRESSION_STATE_ATTR
from examples.torch.common.model_loader import MODEL_STATE_ATTR


def save_checkpoint(model, compression_ctrl, optimizer, epoch, miou, config):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - compression_ctrl (``PTCompressionAlgorithmController``): The controller containing compression state to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - compression_scheduler: The compression scheduler associated with the model
    - config: Model config".

    Returns:
        The path to the saved checkpoint.
    """
    name = config.name
    save_dir = config.checkpoint_save_dir

    assert os.path.isdir(save_dir), 'The directory "{0}" doesn\'t exist.'.format(save_dir)

    # Save model
    checkpoint_path = os.path.join(save_dir, name) + "_last.pth"

    checkpoint = {
        "epoch": epoch,
        "miou": miou,
        MODEL_STATE_ATTR: model.state_dict(),
        COMPRESSION_STATE_ATTR: compression_ctrl.get_compression_state(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
