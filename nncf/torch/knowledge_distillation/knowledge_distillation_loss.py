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

from functools import partial
from functools import reduce
from typing import Any, Callable, Optional

import torch
from torch import nn

from nncf.common.logging import nncf_logger
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.nested_objects_traversal import NestedObjectIndex
from nncf.torch.nncf_network import NNCFNetwork


class KnowledgeDistillationLoss(PTCompressionLoss):
    """
    Doesn't not directly compute knowledge distillation loss values but has access to them through
    KnowledgeDistillationLossHandler. Notice that knowledge distillation loss is computed between results of original
    model and compressed model inferences with latest inputs. Provides KnowledgeDistillationLossHandler with kd original
    model (to distill from), storage device and function to calculate knowledge distillation loss.
    """

    def __init__(
        self, target_model: NNCFNetwork, original_model: nn.Module, kd_type: str, scale: float, temperature: float
    ):
        super().__init__()
        original_model.train()
        if kd_type == "softmax":

            def kd_loss_fn(teacher_output: torch.Tensor, student_output: torch.Tensor):
                if len(student_output.shape) != 2 or len(teacher_output.shape) != 2:
                    nncf_logger.debug(
                        f"Incompatible number of dimensions of the model output tensor for softmax KD "
                        f"(student - {student_output.shape}, "
                        f"teacher - {teacher_output.shape}, "
                        f"number of dims for both should be == 2) - ignoring!"
                    )
                    return torch.zeros([1]).to(student_output.device)
                return (
                    scale
                    * -(
                        nn.functional.log_softmax(student_output / temperature, dim=1)
                        * nn.functional.softmax(teacher_output / temperature, dim=1)
                    ).mean()
                    * (student_output.shape[1] * temperature * temperature)
                )

        else:

            def kd_loss_fn(teacher_output: torch.Tensor, student_output: torch.Tensor):
                mse = torch.nn.MSELoss()
                if len(teacher_output.shape) < 2:
                    nncf_logger.debug(
                        f"Incompatible number of dimensions of the model output tensor for MSE KD "
                        f"(student - {student_output.shape}, "
                        f"teacher - {teacher_output.shape}, "
                        f"number of dims {len(student_output.shape)} should be > 1) (most likely loss) - ignoring!"
                    )
                    return torch.zeros([1]).to(student_output.device)
                return scale * mse(teacher_output, student_output)

        self._kd_loss_handler = target_model.nncf.create_knowledge_distillation_loss_handler(
            original_model, partial(KnowledgeDistillationLoss._calculate, kd_loss_fn=kd_loss_fn)
        )

    @staticmethod
    def _calculate(
        compressed_model_outputs: Any,
        orig_model_outputs: Any,
        kd_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Calculates knowledge distillation loss value from compressed_model_outputs and orig_model_outputs. First uses
        nested_object_paths_generator to unpack input containers and numerate contents inside them.
        Than checks compressed_model_outputs unpacked container for loss tensors (requires_grad=True)
        and maps extracted structure of loss tensors to orig_model_outputs.
        Finally computes knowledge distillation loss with extracted loss tensors.

        :param compressed_model_outputs: Output tensors of compressed model can be any type of container with
            deterministic traversal.
        :param orig_model_outputs: Output tensors of original model (used for distillation) can be any type of
            container with deterministic traversal.
        :return: knowledge distillation loss value
        """

        compressed_model_outputs_nested_obj_indexing = NestedObjectIndex([compressed_model_outputs])
        orig_model_outputs_nested_obj_indexing = NestedObjectIndex([orig_model_outputs])
        compressed_model_loss_outputs_nested_obj_indexing = list(
            filter(
                lambda x: KnowledgeDistillationLoss._is_loss(x.getter()),
                compressed_model_outputs_nested_obj_indexing.get_flat_nested_obj_indexing(),
            )
        )
        compressed_model_loss_outputs = list(
            map(lambda x: x.getter(), compressed_model_loss_outputs_nested_obj_indexing)
        )

        def match_fn(obj):
            for x in compressed_model_loss_outputs_nested_obj_indexing:
                if x.path == obj.path:
                    return True
            return False

        orig_model_loss_outputs = list(
            map(
                lambda x: x.getter(),
                filter(match_fn, orig_model_outputs_nested_obj_indexing.get_flat_nested_obj_indexing()),
            )
        )

        if len(orig_model_loss_outputs) != len(compressed_model_loss_outputs):
            nncf_logger.warning(
                f"KD: mismatch in the number of detected loss tensors in return value between original "
                f"and compressed models;\n"
                f"original has {len(orig_model_loss_outputs)} loss tensors,\n"
                f"compressed has {len(compressed_model_loss_outputs)}"
            )
        if not orig_model_loss_outputs:
            nncf_logger.warning("KD: no loss outputs detected in original model, knowledge distillation not possible")
            return None
        if not compressed_model_loss_outputs:
            nncf_logger.warning("KD: no loss outputs detected in compressed model, knowledge distillation not possible")
            return None
        return reduce(
            lambda kd_loss, loss_tensors: kd_loss + kd_loss_fn(loss_tensors[0], loss_tensors[1]),
            zip(orig_model_loss_outputs, compressed_model_loss_outputs),
            torch.zeros([], device=orig_model_loss_outputs[0].device),
        )

    @staticmethod
    def _is_loss(obj):
        if not isinstance(obj, torch.Tensor):
            return False
        if obj.requires_grad:
            return True
        return False

    def forward(self) -> torch.Tensor:
        """
        Gets knowledge distillation loss values from KnowledgeDistillationLossHandler, averages them in case of
        DataParallel execution (loss values for mini-batches) and frees up KnowledgeDistillationLossHandler loss values
        storage space.

        :return: Differentiable knowledge distillation loss value
        """
        loss = self._kd_loss_handler.get_kd_loss()
        for idx, _ in enumerate(loss):
            loss[idx] = loss[idx].unsqueeze(0)
        output = torch.cat(loss).mean()
        return output

    def statistics(self, quickly_collected_only=False):
        return {}
