# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, TypeVar

from torch import nn

import nncf
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_PT_CATEGORY
from nncf.telemetry.extractors import FunctionCallTelemetryExtractor
from nncf.torch.function_hook import is_wrapped as pt2_is_wrapped
from nncf.torch.function_hook import wrap_model as pt2_wrap_model
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.function_hook.serialization import get_config as pt2_get_config
from nncf.torch.function_hook.serialization import load_from_config as pt2_load_from_config

TModel = TypeVar("TModel", bound=nn.Module)


@tracked_function(
    NNCF_PT_CATEGORY,
    [
        FunctionCallTelemetryExtractor("nncf.torch.wrap_model"),
    ],
)
def wrap_model(
    model: TModel,
    example_input: Any,
    trace_parameters: bool = False,
) -> GraphModelWrapper:
    """
    Wraps a PyTorch model to the NNCFNetwork class.

    This function dynamically extends the instance of PyTorch model with NNCF-enabling functionality.

    :param model: PyTorch model.
    :param example_input: An example input that will be used for model tracing. A tuple is interpreted
        as an example input of a set of non keyword arguments, and a dict as an example input of a set
        of keywords arguments.
    :param trace_parameters: Whether to trace model parameters. Default is False.
    :return: A model wrapped by GraphModelWrapper if experimental PyTorch model tracing is enabled.
    """
    if not trace_parameters:
        msg = "The 'trace_parameters=False' option is not supported in the experimental tracing mode."
        raise nncf.InternalError(msg)

    if not pt2_is_wrapped(model):
        model = pt2_wrap_model(model)
    wrapped_model = GraphModelWrapper(model, example_input=example_input)
    return wrapped_model


def is_wrapped_model(model: Any) -> bool:
    """
    Check that the model was wrapped by NNCFNetwork or GraphModelWrapper.

    :param model: A model.
    :return: True if the model is wrapped, False otherwise.
    """
    from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper

    return isinstance(model, GraphModelWrapper)


@tracked_function(
    NNCF_PT_CATEGORY,
    [
        FunctionCallTelemetryExtractor("nncf.torch.load_from_config"),
    ],
)
def load_from_config(model: nn.Module, config: dict[str, Any], example_input: Optional[Any] = None) -> nn.Module:
    """
    Wraps given model and recovers additional modules from given config.
    Does not recover additional modules weights as they are located in a corresponded state_dict.

    :param model: PyTorch model.
    :param config: NNCNetwork config.
    :param example_input: An example input that will be used for model tracing. A tuple is interpreted
        as an example input of a set of non keyword arguments, and a dict as an example input of a set
        of keywords arguments. Required with enabled legacy tracing mode.
    :return: Wrapped model with additional modules recovered from given config.
    """
    return pt2_load_from_config(model, config)


@tracked_function(
    NNCF_PT_CATEGORY,
    [
        FunctionCallTelemetryExtractor("nncf.torch.get_config"),
    ],
)
def get_config(model: nn.Module) -> dict[str, Any]:
    """
    Returns the configuration object of the compressed model.

    :param model: The compressed model.
    :return: The configuration object of the compressed model.
    """
    return pt2_get_config(model)
