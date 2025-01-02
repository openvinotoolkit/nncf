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
from collections import OrderedDict
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from nncf.common.logging import nncf_logger
from nncf.common.utils.decorators import skip_if_dependency_unavailable
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.adjust_padding import add_adjust_padding_nodes
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.torch.quantization.precision_init.definitions import QConfigSequenceForHAWQToEvaluate
from nncf.torch.quantization.precision_init.perturbations import PerturbationObserver
from nncf.torch.quantization.precision_init.perturbations import Perturbations
from nncf.torch.quantization.precision_init.traces_order import TracesPerLayer
from nncf.torch.utils import get_all_modules_by_type


class HAWQDebugger:
    def __init__(
        self,
        weight_qconfig_sequences_in_trace_order: List["QConfigSequenceForHAWQToEvaluate"],
        perturbations: Perturbations,
        weight_observers_for_each_covering_configuration: List[List[PerturbationObserver]],
        traces_per_layer: TracesPerLayer,
        bitwidths: List[int],
    ):
        self._weight_qconfig_sequences_in_trace_order = weight_qconfig_sequences_in_trace_order
        self._num_weights = len(traces_per_layer.traces_order)
        self._perturbations = perturbations

        from nncf.common.utils.debug import DEBUG_LOG_DIR

        self._dump_dir = Path(DEBUG_LOG_DIR) / Path("hawq_dumps")
        self._dump_dir.mkdir(parents=True, exist_ok=True)

        self._traces_order = traces_per_layer.traces_order
        self._traces_per_layer = traces_per_layer.get_all()

        num_of_weights = []
        norm_of_weights = []
        for i in range(self._num_weights):
            trace_index = self._traces_order.get_execution_index_by_traces_index(i)
            num_of_weights.append(weight_observers_for_each_covering_configuration[0][trace_index].get_numels())
            norm_of_weights.append(weight_observers_for_each_covering_configuration[0][trace_index].get_input_norm())
        self._num_weights_per_layer = torch.Tensor(num_of_weights)
        self._norm_weights_per_layer = torch.Tensor(norm_of_weights)

        bits_in_megabyte = 2**23
        self._model_sizes = []
        for qconfig_sequence in self._weight_qconfig_sequences_in_trace_order:
            size = (
                torch.sum(
                    torch.Tensor([qconfig.num_bits for qconfig in qconfig_sequence]) * self._num_weights_per_layer
                ).item()
                / bits_in_megabyte
            )
            self._model_sizes.append(size)
        self._bitwidths = bitwidths

    @staticmethod
    def get_all_quantizers_per_full_scope(model):
        all_quantizations = OrderedDict()
        for class_type in QUANTIZATION_MODULES.registry_dict.values():
            quantization_type = class_type.__name__
            all_quantizations.update(
                get_all_modules_by_type(
                    model.nncf.get_compression_modules_by_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER),
                    quantization_type,
                )
            )
            all_quantizations.update(get_all_modules_by_type(model, quantization_type))
        all_quantizations = OrderedDict(sorted(all_quantizations.items(), key=lambda x: str(x[0])))
        return all_quantizations

    @skip_if_dependency_unavailable(dependencies=["matplotlib.pyplot"])
    def dump_avg_traces(self):
        import matplotlib.pyplot as plt

        dump_file = os.path.join(self._dump_dir, "avg_traces_per_layer")
        torch.save(self._traces_per_layer, dump_file)
        fig = plt.figure()
        fig.suptitle("Average Hessian Trace")
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale("log")
        ax.set_xlabel("weight quantizers")
        ax.set_ylabel("average hessian trace")
        ax.plot(self._traces_per_layer.cpu().numpy())
        plt.savefig(dump_file)

    @skip_if_dependency_unavailable(dependencies=["matplotlib.pyplot"])
    def dump_metric_MB(self, metric_per_qconfig_sequence: List[Tensor]):
        import matplotlib.pyplot as plt

        list_to_plot = [cm.item() for cm in metric_per_qconfig_sequence]
        fig = plt.figure()
        fig.suptitle("Pareto Frontier")
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale("log")
        ax.set_xlabel("Model Size (MB)")
        ax.set_ylabel("Metric value (total perturbation)")
        ax.scatter(self._model_sizes, list_to_plot, s=20, facecolors="none", edgecolors="r")
        cm = torch.Tensor(metric_per_qconfig_sequence)
        cm_m = cm.median().item()
        qconfig_index = metric_per_qconfig_sequence.index(cm_m)
        ms_m = self._model_sizes[qconfig_index]
        ax.scatter(ms_m, cm_m, s=30, facecolors="none", edgecolors="b", label="median from all metrics")
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, "Pareto_Frontier"))
        nncf_logger.debug(
            f"Distribution of HAWQ metrics: "
            f"min_value={cm.min().item():.3f}, "
            f"max_value={cm.max().item():.3f}, "
            f"median_value={cm_m:.3f}, "
            f"median_index={qconfig_index}, "
            f"total_number={len(metric_per_qconfig_sequence)}"
        )

    @skip_if_dependency_unavailable(dependencies=["matplotlib.pyplot"])
    def dump_metric_flops(
        self, metric_per_qconfig_sequence: List[Tensor], flops_per_config: List[float], choosen_qconfig_index: int
    ):
        import matplotlib.pyplot as plt

        list_to_plot = [cm.item() for cm in metric_per_qconfig_sequence]
        fig = plt.figure()
        fig.suptitle("Pareto Frontier")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Compression ratio: total INT8 Bits Complexity / total MIXED INT Bits Complexity")
        ax.set_ylabel("Metric value (total perturbation)")
        ax.scatter(flops_per_config, list_to_plot, s=10, alpha=0.3)  # s=20, facecolors='none', edgecolors='r')
        flops_per_config = [torch.Tensor([v]) for v in flops_per_config]
        cm = torch.Tensor(flops_per_config)
        cm_m = cm.median().item()
        configuration_index = flops_per_config.index(cm_m)
        ms_m = metric_per_qconfig_sequence[configuration_index].item()
        ax.scatter(cm_m, ms_m, s=30, facecolors="none", edgecolors="b", label="median from all metrics")
        cm_c = metric_per_qconfig_sequence[choosen_qconfig_index].item()
        fpc_c = flops_per_config[choosen_qconfig_index].item()
        ax.scatter(fpc_c, cm_c, s=30, facecolors="none", edgecolors="r", label="chosen config")

        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, "Pareto_Frontier_compress_ratio"))

    @skip_if_dependency_unavailable(dependencies=["matplotlib.pyplot"])
    def dump_density_of_quantization_noise(self):
        noise_per_config: List[Tensor] = []
        for qconfig_sequence in self._weight_qconfig_sequences_in_trace_order:
            qnoise = 0
            for i in range(self._num_weights):
                execution_index = self._traces_order.get_execution_index_by_traces_index(i)
                qnoise += self._perturbations.get(layer_id=execution_index, qconfig=qconfig_sequence[i])
            noise_per_config.append(qnoise)

        list_to_plot = [cm.item() for cm in noise_per_config]
        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.suptitle("Density of quantization noise")
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale("log")
        ax.set_xlabel("Blocks")
        ax.set_ylabel("Noise value")
        ax.scatter(self._model_sizes, list_to_plot, s=20, alpha=0.3)
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, "Density_of_quantization_noise"))

    @skip_if_dependency_unavailable(dependencies=["matplotlib.pyplot"])
    def dump_perturbations_ratio(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.suptitle("Quantization noise vs Average Trace")
        ax = fig.add_subplot(2, 1, 1)
        ax.set_xlabel("Blocks")
        ax.set_yscale("log")
        perturbations_per_layer_id = list(self._perturbations.get_all().values())
        perturb = []
        max_bitwidths = []
        for perturbations_for_all_observed_qconfig_sequence_in_current_layer in perturbations_per_layer_id:
            qconfig_sequence = perturbations_for_all_observed_qconfig_sequence_in_current_layer.keys()
            max_bitwidth_qconfig = max(qconfig_sequence, key=lambda x: x.num_bits)
            perturb.append(perturbations_for_all_observed_qconfig_sequence_in_current_layer[max_bitwidth_qconfig])
            max_bitwidths.append(max_bitwidth_qconfig.num_bits)
        ax.plot(
            [
                (p / m / n).cpu().numpy()
                for p, m, n in zip(perturb, self._num_weights_per_layer, self._norm_weights_per_layer)
            ],
            label="normalized n-bit noise",
        )
        ax.plot([x.cpu().numpy() for x in perturb], label="n-bit noise")
        ax.plot(max_bitwidths, label="n")
        ax.plot(self._traces_per_layer.cpu().numpy(), label="trace")
        ax.plot([(n * p).cpu().numpy() for n, p in zip(self._traces_per_layer, perturb)], label="trace * noise")
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, "Quantization_noise_vs_Average_Trace"))

    def dump_bitwidth_graph(
        self,
        algo_ctrl: "QuantizationController",  # noqa: F821
        model: NNCFNetwork,
        groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
    ):
        from nncf.torch.quantization.precision_init.bitwidth_graph import BitwidthGraph

        bw_graph = BitwidthGraph(algo_ctrl, model, groups_of_adjacent_quantizers).get()
        nx_graph = add_adjust_padding_nodes(bw_graph, model)
        write_dot_graph(nx_graph, self._dump_dir / Path("bitwidth_graph.dot"))
