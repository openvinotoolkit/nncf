# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
from typing import Any, List, Optional

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.common import CONSTANT_OPERATIONS
from nncf.openvino.graph.metatypes.common import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.metatypes.common import QUANTIZABLE_OPERATIONS
from nncf.openvino.graph.metatypes.common import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.openvino.graph.metatypes.common import SHAPEOF_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConcatMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.backend import AsyncPreparedModel


def compile_model(model: ov.Model, done_queue: multiprocessing.Queue) -> None:
    compiled_model = ov.Core().compile_model(model, "CPU")
    model_stream = compiled_model.export_model()
    done_queue.put(model_stream)


class OVAsyncPreparedModel(AsyncPreparedModel):
    def __init__(self, proc: multiprocessing.Process, done_queue: multiprocessing.Queue):
        self.proc = proc
        self.done_queue = done_queue

    def get(self, timeout=None) -> ov.CompiledModel:
        try:
            model_stream = self.done_queue.get(timeout=timeout)
        except multiprocessing.TimeoutError as ex:
            raise TimeoutError() from ex
        return ov.Core().import_model(model_stream, "CPU")


class OVAccuracyControlAlgoBackend(AccuracyControlAlgoBackend):
    """
    Implementation of the `AccuracyControlAlgoBackend` for OpenVINO backend.
    """

    # Metatypes

    @staticmethod
    def get_quantizer_metatypes() -> List[OVOpMetatype]:
        return FAKE_QUANTIZE_OPERATIONS

    @staticmethod
    def get_const_metatypes() -> List[OVOpMetatype]:
        return CONSTANT_OPERATIONS

    @staticmethod
    def get_quantizable_metatypes() -> List[OVOpMetatype]:
        return QUANTIZABLE_OPERATIONS

    @staticmethod
    def get_quantize_agnostic_metatypes() -> List[OVOpMetatype]:
        return QUANTIZE_AGNOSTIC_OPERATIONS + [OVConcatMetatype]

    @staticmethod
    def get_shapeof_metatypes() -> List[OVOpMetatype]:
        return SHAPEOF_OPERATIONS

    # Manipulations with bias value and weights

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def is_node_with_weight(node: NNCFNode) -> bool:
        return node.metatype in GENERAL_WEIGHT_LAYER_METATYPES and isinstance(node.layer_attributes, OVLayerAttributes)

    @staticmethod
    def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> np.ndarray:
        return get_bias_value(node_with_bias, nncf_graph, model)

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: ov.Model, port_id: int) -> np.ndarray:
        return get_weight_value(node_with_weight, model, port_id)

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        return node.layer_attributes.get_const_port_ids()

    @staticmethod
    def get_model_size(model: ov.Model) -> int:
        model_size = 0
        for op in model.get_ops():
            if op.get_type_name() == "Constant":
                model_size += op.data.nbytes

        return model_size

    # Preparation of model

    @staticmethod
    def prepare_for_inference(model: ov.Model) -> Any:
        return ov.compile_model(model)

    @staticmethod
    def prepare_for_inference_async(model: ov.Model) -> Any:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=compile_model, args=(model, queue))
        p.start()
        return OVAsyncPreparedModel(p, queue)
