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
from torch.export import ExportedProgram
from transformers import GenerationConfig
from transformers import GenerationMixin
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig


class FXAutoModelForCausalLM(OptimizedModel, GenerationMixin):
    def __init__(
        self,
        model: torch.fx.GraphModule,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        device: str = "cpu",
        compile: bool = True,
    ) -> None:
        super().__init__(model, config)
        if compile:
            self.model = torch.compile(model, backend="openvino", options={"aot_autograd": True})
        self.generation_config = generation_config
        self.main_input_name = "input_ids"
        self._device = device.upper()

    @property
    def device(self) -> torch.device:
        return torch.device(self._device.lower())

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        cache_position = kwargs["cache_position"]
        past_len = cache_position[0]
        if past_len < input_ids.shape[1]:
            input_ids = input_ids[:, past_len:]

        return {"input_ids": input_ids, "cache_position": cache_position}

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        logits = self.model(input_ids, cache_position)
        return CausalLMOutputWithPast(logits=logits)

    def _save_pretrained(self, save_directory: str) -> None:
        pass

    def can_generate(self) -> bool:
        return True

    def _supports_default_dynamic_cache(self) -> bool:
        return False


class TorchExportableModuleWithStaticCacheDynamicShape(TorchExportableModuleWithStaticCache):
    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        abc = cache_position.unsqueeze(0)
        outs = self.model(
            input_ids=input_ids,
            position_ids=abc,
            cache_position=cache_position,
            past_key_values=self.static_cache,
            use_cache=True,
        )
        return outs.logits


@torch.no_grad()
def convert_and_export_with_cache(model: PreTrainedModel) -> tuple[ExportedProgram, LlamaConfig, GenerationConfig]:
    """
    Convert a `PreTrainedModel` into an exportable module and export it using `torch.export`
    or `torch._export.capture_pre_autograd_graph`.

    :param model: Model to be exported.
    :return: A tuple iwth the Exported Program, config for the LLM, Generation config of the LLM
    """

    example_input_ids = torch.ones(1, 8, dtype=torch.long)
    example_cache_position = torch.arange(0, 8, dtype=torch.long)
    model_config = None
    gen_config = None
    model.generation_config.cache_implementation = "static"
    model.generation_config.cache_config = {"batch_size": 1, "max_cache_len": 512}
    model.generation_config.max_new_tokens = 100
    gen_config = model.generation_config
    model_config = model.config
    model = TorchExportableModuleWithStaticCacheDynamicShape(model)

    dynamic_shapes = {"input_ids": {1: torch.export.Dim.DYNAMIC}, "cache_position": {0: torch.export.Dim.DYNAMIC}}

    exported_program = torch.export.export_for_training(
        model,
        args=(
            example_input_ids,
            example_cache_position,
        ),
        dynamic_shapes=dynamic_shapes,
    ).run_decompositions(decomp_table={})
    return exported_program, model_config, gen_config
