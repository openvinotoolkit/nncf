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

import tempfile
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torchvision.transforms.functional as F
from torch import nn

from tests.torch.models_hub_test.common import BaseTestModel
from tests.torch.models_hub_test.common import ExampleType
from tests.torch.models_hub_test.common import ModelInfo
from tests.torch.models_hub_test.common import get_model_params
from tests.torch.models_hub_test.common import idfn

MODEL_LIST_FILE = Path(__file__).parent / "torchvision_models.txt"


def get_video():
    """
    Download video and return frames.
    Using free video from pexels.com, credits go to Pavel Danilyuk.
    Initially used in https://pytorch.org/vision/stable/auto_examples/plot_optical_flow.html
    """
    from pathlib import Path
    from urllib.request import urlretrieve

    from torchvision.io import read_video

    video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "basketball.mp4"
        _ = urlretrieve(video_url, video_path)

        frames, _, _ = read_video(str(video_path), output_format="TCHW")
    return frames


def prepare_frames_for_raft(name, frames1, frames2):
    w = torch.hub.load("pytorch/vision", "get_model_weights", name=name, skip_validation=True).DEFAULT
    img1_batch = torch.stack(frames1)
    img2_batch = torch.stack(frames2)
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    img1_batch, img2_batch = w.transforms()(img1_batch, img2_batch)
    return (img1_batch, img2_batch)


class TestTorchHubModel(BaseTestModel):
    def load_model(self, model_name: str) -> Tuple[nn.Module, ExampleType]:
        torch.manual_seed(0)
        m = torch.hub.load("pytorch/vision", model_name, weights=None, skip_validation=True)
        m.eval()
        if model_name == "s3d" or any([m in model_name for m in ["swin3d", "r3d_18", "mc3_18", "r2plus1d_18"]]):
            example = (torch.randn([1, 3, 224, 224, 224]),)

        elif "mvit" in model_name:
            # 16 frames from video
            example = (torch.randn(1, 3, 16, 224, 224),)
        elif "raft" in model_name:
            frames = get_video()
            example = prepare_frames_for_raft(model_name, [frames[100], frames[150]], [frames[101], frames[151]])
        else:
            example = (torch.randn(1, 3, 224, 224),)
        return m, example

    @pytest.mark.parametrize("model_info", get_model_params(MODEL_LIST_FILE), ids=idfn)
    def test_nncf_wrap(self, model_info: ModelInfo):
        self.nncf_wrap(model_info.model_name)
