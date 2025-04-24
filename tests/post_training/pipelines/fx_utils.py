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

import torch
import torch.fx
from optimum.modeling_base import OptimizedModel
from transformers import GenerationConfig
from transformers import GenerationMixin
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import StaticCacheConfig
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast


class FXAutoModelForCausalLM(OptimizedModel, GenerationMixin):
    def __init__(
        self,
        model: torch.fx.GraphModule,
        config: PretrainedConfig,
        device: str = "cpu",
        compile: bool = True,
        backend: str = "openvino",
        dtype=torch.float32,
    ):
        super().__init__(model, config)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.compile = compile
        self.backend = backend
        self._device = device.upper()
        self._dtype = dtype
        self._cached_prefill_input_ids = None
        self._cached_cache_position = None
        self.compiled_decode = False
        self.compiled_prefill = False
        if self.compile:
            self.prefill = None
            self.decode_one_token = None

    def get_openvino_backend_options(self) -> dict:
        return {
            "device": self._device,
            "aot_autograd": True,
        }

    def get_prefill(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if (
            self.prefill is None
            or self._cached_prefill_input_ids.shape != input_ids.shape
            or self._cached_cache_position.shape != cache_position.shape
        ):
            self._cached_prefill_input_ids = input_ids
            self._cached_cache_position = cache_position
            self.prefill = self.model
            self.prefill.forward = torch.compile(
                self.prefill.forward,
                backend="openvino",
                options=self.get_openvino_backend_options(),
            )
        return self.prefill

    def get_decode_one_token(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.decode_one_token is None:
            self.decode_one_token = self.model
            self.decode_one_token.forward = torch.compile(
                self.decode_one_token.forward,
                backend="openvino",
                options=self.get_openvino_backend_options(),
            )
        return self.decode_one_token

    def infer_prefill(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.compile:
            _ = self.get_prefill(input_ids, cache_position)(input_ids, cache_position)
        else:
            self.model(input_ids, cache_position)

    def infer_decode_one_token(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        if self.compile:
            _ = self.get_decode_one_token(input_ids, cache_position)(input_ids, cache_position)
        else:
            self.model(input_ids, cache_position)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device.lower())

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        cache_position = kwargs["cache_position"]
        past_len = cache_position[0]
        if past_len < input_ids.shape[1]:
            input_ids = input_ids[:, past_len:]

        return {"input_ids": input_ids, "cache_position": cache_position}

    def _save_pretrained(self, save_directory):
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.compile:
            if input_ids.shape[1] == 1:
                logits = self.get_decode_one_token(input_ids, cache_position)(input_ids, cache_position)
            else:
                logits = self.get_prefill(input_ids, cache_position)(input_ids, cache_position)
        else:
            logits = self.model(input_ids, cache_position)
        return CausalLMOutputWithPast(logits=logits)

    def can_generate(self):
        return True

    def _supports_default_dynamic_cache(self) -> bool:
        return False


class TorchExportableModuleWithStaticCacheDynamicShape(TorchExportableModuleWithStaticCache):
    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        outs = self.model(
            input_ids=input_ids,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outs.logits


def convert_and_export_with_cache(model: PreTrainedModel, re_export=False):
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`
    or `torch._export.capture_pre_autograd_graph`.
    """

    with torch.no_grad():
        example_input_ids = torch.ones(1, 8, dtype=torch.long)
        example_cache_position = torch.arange(0, 8, dtype=torch.long)
        model_config = None
        if not re_export:
            model.generation_config.cache_implementation = "static"
            model.generation_config.cache_config = StaticCacheConfig(batch_size=1, max_cache_len=512)
            model(example_input_ids)
            model_config = model.config
            model = TorchExportableModuleWithStaticCacheDynamicShape(model)

        sequence_length = torch.export.Dim("sequence_length", min=1, max=512)
        dynamic_shapes = {"input_ids": {1: sequence_length}, "cache_position": {0: sequence_length}}

        exported_program = torch.export.export_for_training(
            model,
            args=(
                example_input_ids,
                example_cache_position,
            ),
            dynamic_shapes=dynamic_shapes,
        ).run_decompositions(decomp_table={})
        return exported_program, model_config
