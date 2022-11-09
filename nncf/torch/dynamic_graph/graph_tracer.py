"""
 Copyright (c) 2019-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from collections import OrderedDict
from typing import Callable, Any, List, Optional
from copy import deepcopy

import torch

from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.utils import get_model_device


class ModelInputInfo:
    FILLER_TYPE_ONES = "ones"
    FILLER_TYPE_ZEROS = "zeros"
    FILLER_TYPE_RANDOM = "random"
    FILLER_TYPES = [FILLER_TYPE_ONES, FILLER_TYPE_ZEROS, FILLER_TYPE_RANDOM]

    def __init__(self, shape: List[int], type_str: str = "float", keyword=None, filler=None):
        self.shape = shape
        self.type = self._string_to_torch_type(type_str)
        self.keyword = keyword
        if filler is None:
            self.filler = self.FILLER_TYPE_ONES
        else:
            self.filler = filler
            if self.filler not in self.FILLER_TYPES:
                raise RuntimeError("Unknown input filler type: {}".format(filler))

    @staticmethod
    def _string_to_torch_type(string):
        if string == "long":
            return torch.long
        return torch.float32

    @staticmethod
    def torch_type_to_string(dtype: torch.dtype):
        if dtype is torch.long:
            return "long"
        return "float"

    def is_integer_input(self):
        return self.type != torch.float32

    def __eq__(self, other):
        return self.type == other.type and self.keyword == other.keyword


def create_input_infos(config) -> Optional[List[ModelInputInfo]]:
    input_infos = config.get("input_info")
    if input_infos is None:
        return input_infos
    if isinstance(input_infos, dict):
        return [ModelInputInfo(input_infos.get("sample_size"),
                               input_infos.get("type"),
                               input_infos.get("keyword"),
                               input_infos.get("filler")), ]
    if isinstance(input_infos, list):
        return [ModelInputInfo(info_dict.get("sample_size"),
                               info_dict.get("type"),
                               info_dict.get("keyword"),
                               info_dict.get("filler")) for info_dict in input_infos]
    raise RuntimeError("Invalid input_infos specified in config - should be either dict or list of dicts")


def create_mock_tensor(input_info: ModelInputInfo, device: str):
    args = {"size": input_info.shape, "dtype": input_info.type, "device": device}
    if input_info.filler == ModelInputInfo.FILLER_TYPE_ZEROS:
        return torch.zeros(**args)
    if input_info.filler == ModelInputInfo.FILLER_TYPE_ONES:
        return torch.ones(**args)
    if input_info.filler == ModelInputInfo.FILLER_TYPE_RANDOM:
        return torch.rand(**args)
    raise RuntimeError


class GraphTracer:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def trace_graph(self, model: torch.nn.Module, context_to_use: Optional['TracingContext'] = None,
                    as_eval: bool = False) -> DynamicGraph:
        sd = deepcopy(model.state_dict())

        from nncf.torch.dynamic_graph.context import TracingContext #pylint: disable=cyclic-import
        if context_to_use is None:
            context_to_use = TracingContext()

        context_to_use.enable_trace_dynamic_graph()
        from nncf.torch.utils import training_mode_switcher #pylint: disable=cyclic-import
        with context_to_use as _ctx:
            _ctx.base_module_thread_local_replica = model
            with torch.no_grad():
                if as_eval:
                    with training_mode_switcher(model, is_training=False):
                        self.custom_forward_fn(model)
                else:
                    self.custom_forward_fn(model)
        model.load_state_dict(sd)

        if isinstance(model, PostGraphBuildActing):
            model.post_build_graph_actions()
        context_to_use.disable_trace_dynamic_graph()
        return context_to_use.graph


class PostGraphBuildActing:
    def post_build_graph_actions(self):
        pass


def create_dummy_forward_fn(input_infos: List[ModelInputInfo], with_input_tracing=False,
                            wrap_inputs_fn=None,
                            wrap_outputs_fn=None,
                            with_output_tracing=False):

    def default_dummy_forward_fn(model):
        from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk #pylint: disable=cyclic-import
        from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk #pylint: disable=cyclic-import
        from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors #pylint: disable=cyclic-import

        device = get_model_device(model)
        args_list = [create_mock_tensor(info, device) for info in input_infos if info.keyword is None]
        kwargs = OrderedDict()
        for info in input_infos:
            if info.keyword is not None:
                kwargs[info.keyword] = create_mock_tensor(info, device)
        args = tuple(args_list)

        if with_input_tracing:
            if wrap_inputs_fn is None:
                # We control the input argument structure w.r.t. tensors
                # - a simple objwalk application should be sufficient in this simple case.
                # For more control, wrap_inputs_fn is used when this is used in NNCFNetwork
                # which is guaranteed to be the same as during the actual NNCFNetwork.forward
                args, kwargs = wrap_nncf_model_inputs_with_objwalk(args, kwargs)
            else:
                args, kwargs = wrap_inputs_fn(args, kwargs)
        retval = model(*args, **kwargs)
        if with_output_tracing:
            retval = replicate_same_tensors(retval)
            if wrap_outputs_fn is not None:
                return wrap_outputs_fn(retval)
            return wrap_nncf_model_outputs_with_objwalk(retval)
        return retval
    return default_dummy_forward_fn
