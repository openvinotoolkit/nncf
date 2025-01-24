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
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch

from nncf.common.logging import nncf_logger
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.dynamic_graph.io_handling import LoaderInputInfo
from nncf.torch.dynamic_graph.io_handling import ModelInputInfo
from nncf.torch.dynamic_graph.wrappers import wrap_parameters
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_multidevice


class GraphTracer:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def trace_graph(
        self,
        model: torch.nn.Module,
        context_to_use: Optional[TracingContext] = None,
        as_eval: bool = False,
        trace_parameters: bool = False,
    ) -> DynamicGraph:
        if context_to_use is None:
            context_to_use = TracingContext()

        context_to_use.enable_trace_dynamic_graph()
        from nncf.torch.utils import training_mode_switcher

        with context_to_use as _ctx:
            _ctx.base_module_thread_local_replica = model
            with torch.no_grad():
                if trace_parameters:
                    wrap_parameters(model)

                if as_eval:
                    with training_mode_switcher(model, is_training=False):
                        self.custom_forward_fn(model)
                else:
                    self.custom_forward_fn(model)

        context_to_use.disable_trace_dynamic_graph()
        return context_to_use.graph


T = TypeVar("T")
WrapInputsFnType = Callable[[Tuple, Dict], Tuple[Tuple, Dict]]
WrapOutputsFnType = Callable[[T], T]


def create_dummy_forward_fn(
    input_info: ModelInputInfo,
    with_input_tracing: bool = False,
    wrap_inputs_fn: WrapInputsFnType = None,
    wrap_outputs_fn: WrapOutputsFnType = None,
    with_output_tracing: bool = False,
):
    def default_dummy_forward_fn(model):
        from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors
        from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk
        from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk

        device = None
        if isinstance(input_info, (FillerInputInfo, LoaderInputInfo)):
            if is_multidevice(model):
                nncf_logger.warning(
                    "Multidevice model detected when tracing the model's dynamic graph - will pass example "
                    "inputs to the model as-is without changing their device."
                )
            else:
                device = get_model_device(model)
        args, kwargs = input_info.get_forward_inputs(device)

        if with_input_tracing:
            if wrap_inputs_fn is None:
                # We control the input argument structure w.r.t. tensors if input_info is a FillerInputInfo
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
