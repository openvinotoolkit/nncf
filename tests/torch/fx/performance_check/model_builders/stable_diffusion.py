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

import torch
from diffusers import StableDiffusionPipeline

from tests.torch.fx.performance_check.model_builders.base import BaseModelBuilder


class StableDiffusion2UnetBuilder(BaseModelBuilder):
    def __init__(self):
        latents_shape = (2, 4, 96, 96)
        encoder_hidden_state_shape = (2, 77, 1024)
        time_shape = ()
        self._input_sizes = (latents_shape, time_shape, encoder_hidden_state_shape)
        self._example_input = tuple([torch.ones(shape) for shape in self._input_sizes])

    def build(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        pipe = pipe.to("cpu")
        return pipe.unet.eval()

    def get_example_inputs(self) -> torch.Tensor:
        return (self._example_input,)

    def get_input_sizes(self):
        return self._input_sizes
