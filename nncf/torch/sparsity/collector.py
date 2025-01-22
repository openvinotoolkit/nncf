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

from typing import List

from nncf.common.sparsity.collector import BaseSparseModelStatisticsCollector
from nncf.common.sparsity.collector import WeightDescription
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import SparseModuleInfo


class PTSparseModelStatisticsCollector(BaseSparseModelStatisticsCollector):
    """
    Collects statistics for the sparse NNCFNetwork.
    """

    def __init__(
        self, model: NNCFNetwork, sparse_modules_info: List[SparseModuleInfo], supports_sparse_bias: bool = False
    ):
        """
        Initializes statistics collector of the sparse tf.keras.Model.

        :param model: Sparse model.
        :param sparse_modules_info: List of `SparseModuleInfo`.
        """
        self._model = model
        self._sparse_modules_info = sparse_modules_info
        self._supports_sparse_bias = supports_sparse_bias

    def _collect_weights_descriptions(self) -> List[WeightDescription]:
        weights_descriptions = []
        processed_modules = []

        for minfo in self._sparse_modules_info:
            sparse_weight = minfo.operand.apply_binary_mask(minfo.module.weight)

            weights_descriptions.append(
                WeightDescription(
                    minfo.module_node_name,
                    list(sparse_weight.shape),
                    sparse_weight.count_nonzero().item(),
                    is_sparse=True,
                )
            )

            if hasattr(minfo.module, "bias") and minfo.module.bias is not None:
                bias = minfo.module.bias
                name = f"{minfo.module_node_name}/bias"
                if self._supports_sparse_bias:
                    sparse_bias = minfo.operand.apply_binary_mask(bias, is_bias=True)  # TODO(yujie): breaking changes
                    weights_descriptions.append(
                        WeightDescription(
                            name, list(sparse_bias.shape), sparse_bias.count_nonzero().item(), is_sparse=True
                        )
                    )
                else:
                    weights_descriptions.append(
                        WeightDescription(name, list(bias.shape), bias.count_nonzero().item(), is_sparse=False)
                    )

            processed_modules.append(minfo.module)

        compression_types = tuple(COMPRESSION_MODULES.registry_dict.values())
        modules_to_process = {k: v for k, v in self._model.named_modules() if not k.startswith("_nncf")}
        for module_name, module in modules_to_process.items():
            if isinstance(module, compression_types) or module in processed_modules:
                continue

            for param_name, param in module.named_parameters(recurse=False):
                name = f"{module_name}/{param_name}"
                weights_descriptions.append(
                    WeightDescription(name, list(param.shape), param.count_nonzero().item(), is_sparse=False)
                )

        return weights_descriptions
