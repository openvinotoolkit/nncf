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
import abc
from copy import deepcopy
from functools import partial
from inspect import Parameter
from inspect import Signature
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Type, Union

import torch

import nncf
from nncf import Dataset
from nncf import NNCFConfig
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.common.logging import nncf_logger
from nncf.common.utils.api_marker import api
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.torch.dynamic_graph.context import forward_nncf_trace
from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_tensor
from nncf.torch.utils import is_traced_tensor


@api(canonical_alias="nncf.torch.nncf_model_input")
@register_operator(name=MODEL_INPUT_OP_NAME)
def nncf_model_input(tensor: "torch.Tensor"):
    return tensor


@api(canonical_alias="nncf.torch.nncf_model_output")
@register_operator(name=MODEL_OUTPUT_OP_NAME)
def nncf_model_output(tensor: "torch.Tensor"):
    return tensor


def wrap_nncf_model_inputs_with_objwalk(model_args, model_kwargs):
    model_args = objwalk(model_args, is_tensor, nncf_model_input)
    model_kwargs = objwalk(model_kwargs, is_tensor, nncf_model_input)
    return model_args, model_kwargs


def wrap_nncf_model_outputs_with_objwalk(model_outputs):
    model_outputs = objwalk(model_outputs, is_traced_tensor, nncf_model_output)
    return model_outputs


def replicate_same_tensors(obj: Any) -> Any:
    """
    Required to handle the situation when multiple references to one and the
    same tensor are present in the input. If tensor replication is not done, then
    at runtime one and the same tensor could be wrapped by input/output wrappers twice,
    which will disrupt the traced graph structure and possibly hook calls.
    """
    observed_tensor_object_ids: Set[int] = set()

    def replicate_fn(tensor: torch.Tensor) -> torch.Tensor:
        tensor_object_id = id(tensor)
        if tensor_object_id in observed_tensor_object_ids:
            with forward_nncf_trace():
                return tensor.clone()
        observed_tensor_object_ids.add(tensor_object_id)
        return tensor

    obj = objwalk(obj, is_tensor, replicate_fn)
    return obj


class ModelInputInfo(abc.ABC):
    """
    Provides the model inputs that are subsequently used as example inputs to the forward function of
    the compressed model for purposes of compression (e.g. at the stage of building the dynamic graph, exporting
    to a serialized state etc.)
    """

    @abc.abstractmethod
    def get_forward_inputs(self, device: Optional[Union[str, torch.device]] = None) -> Tuple[Tuple, Dict]:
        """
        Returns the tuple of (args, kwargs) for passing into the compressed model's forward method when necessary.
        The returned arguments should be such that the model's forward with these arguments executes the main
        control flow of the model that needs to be compressed.

        :param device: Optional - a PyTorch string representation of the device that the tensors among the returned
          args, kwargs should be located at.
        """
        pass


class FillerInputElement:
    """
    Represents a single tensor argument (positional or keyword) in the model's example input that is to be generated
    on the fly and filled with a requested type of data filler.
    """

    FILLER_TYPE_ONES = "ones"
    FILLER_TYPE_ZEROS = "zeros"
    FILLER_TYPE_RANDOM = "random"
    FILLER_TYPES = [FILLER_TYPE_ONES, FILLER_TYPE_ZEROS, FILLER_TYPE_RANDOM]

    def __init__(self, shape: List[int], type_str: str = "float", keyword: str = None, filler: str = None):
        """
        :param shape: The shape of the model input tensor.
        :param type_str: The type of the model input tensor - "float" for torch.float32, "long" for torch.long
        :param keyword: Optional - if specified, then this input tensor will be passed as a corresponding keyword
          parameter, and as a positional argument if this parameter is unspecified.
        :param filler: Optional - can be either "ones", "zeros" or "random". The model input tensor will be generated
          with data corresponding to this setting. Default is "ones".
        """
        self.shape = shape
        self.type = self._string_to_torch_type(type_str)
        self.keyword = keyword
        if filler is None:
            self.filler = self.FILLER_TYPE_ONES
        else:
            self.filler = filler
            if self.filler not in self.FILLER_TYPES:
                raise ValueError(f"Unknown input filler type: {filler}")

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

    def __eq__(self, other: "FillerInputElement"):
        return self.type == other.type and self.keyword == other.keyword and self.filler == other.filler

    def get_tensor_for_input(self) -> torch.Tensor:
        if self.filler == FillerInputElement.FILLER_TYPE_ZEROS:
            return torch.zeros(size=self.shape, dtype=self.type)
        if self.filler == FillerInputElement.FILLER_TYPE_ONES:
            return torch.ones(size=self.shape, dtype=self.type)
        if self.filler == FillerInputElement.FILLER_TYPE_RANDOM:
            return torch.rand(size=self.shape, dtype=self.type)
        raise NotImplementedError


class FillerInputInfo(ModelInputInfo):
    """
    An implementation of ModelInputInfo that defines the model input in terms of shapes and types of individual
    tensor args and kwargs of the model's forward method.
    """

    def __init__(self, elements: List[FillerInputElement]):
        super().__init__()
        self.elements = deepcopy(elements)

    @classmethod
    def from_nncf_config(cls, config: NNCFConfig):
        """
        Parses the NNCFConfig's "input_info" field if it is present to determine model input information,
        otherwise raises a RuntimeError. The "input_info" field structure must conform to the NNCF config jsonschema.

        :param config: An NNCFConfig instance.
        :return: FillerInputInfo object initialized according to config.
        """
        input_infos = config.get("input_info")
        if input_infos is None:
            raise nncf.ValidationError("Passed NNCFConfig does not have an 'input_info' field")
        if isinstance(input_infos, dict):
            return FillerInputInfo(
                [
                    FillerInputElement(
                        input_infos.get("sample_size"),
                        input_infos.get("type"),
                        input_infos.get("keyword"),
                        input_infos.get("filler"),
                    )
                ]
            )
        if isinstance(input_infos, list):
            elements: List[FillerInputElement] = []
            for info_dict in input_infos:
                elements.append(
                    FillerInputElement(
                        info_dict.get("sample_size"),
                        info_dict.get("type"),
                        info_dict.get("keyword"),
                        info_dict.get("filler"),
                    )
                )
            return FillerInputInfo(elements)
        raise nncf.ValidationError("Invalid input_infos specified in config - should be either dict or list of dicts")

    def get_forward_inputs(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        args_list = []
        kwargs = {}
        for fe in self.elements:
            tensor = fe.get_tensor_for_input()
            if device is not None:
                tensor = tensor.to(device)
            if fe.keyword is None:
                args_list.append(tensor)
            else:
                kwargs[fe.keyword] = tensor
        return tuple(args_list), kwargs


class ExactInputsInfo(ModelInputInfo):
    """
    An implementation of ModelInputInfo that defines the model input in terms of exact forward args and kwargs.
    """

    def __init__(self, forward_args: Tuple, forward_kwargs: Dict):
        self._forward_args = forward_args
        self._forward_kwargs = forward_kwargs

    def get_forward_inputs(self, device: Optional[Union[str, torch.device]] = None) -> Tuple[Tuple, Dict]:
        if device is None:
            return self._forward_args, self._forward_kwargs
        to_device_fn = partial(torch.Tensor.to, device=device)
        args_copy = deepcopy(self._forward_args)
        kwargs_copy = deepcopy(self._forward_kwargs)
        args_at_device = objwalk(args_copy, is_tensor, to_device_fn)
        kwargs_at_device = objwalk(kwargs_copy, is_tensor, to_device_fn)
        return args_at_device, kwargs_at_device


class ExampleInputInfo(ExactInputsInfo):
    @classmethod
    def from_example_input(cls, example_input: Any) -> "ExampleInputInfo":
        """
        Builds an ExampleInputInfo object based on the example input.

        :param dataset: An nncf.Dataset whose first element will be used as an example model input
        :return: An initialized ExampleInputInfo object.
        """
        if isinstance(example_input, tuple):
            return ExampleInputInfo(example_input, {})
        if isinstance(example_input, dict):
            return ExampleInputInfo(tuple(), example_input)
        return ExampleInputInfo((example_input,), {})

    @classmethod
    def from_nncf_dataset(cls, dataset: Dataset) -> "ExampleInputInfo":
        """
        Checks the first element of the provided nncf.Dataset and builds an ExampleInputInfo object that would
        provide the same input to the model at corresponding compression stages.

        :param dataset: An nncf.Dataset whose first element will be used as an example model input
        :return: An initialized ExampleInputInfo object.
        """
        example_input = next(iter(dataset.get_inference_data()))
        return cls.from_example_input(example_input)


class LoaderInputInfo(ExactInputsInfo):
    @classmethod
    def from_nncf_config_dataloaders(cls, config: NNCFConfig) -> Optional["LoaderInputInfo"]:
        """
        Examines the user-provided structures registered with the NNCFConfig instance used for compression to find
        structures that contain a dataloader. The dataloader's first element is used to provide an example input to
        the compressed model.

        :param config: An nncf.NNCFConfig instance. Must have at least one NNCFExtraConfigStruct attached that can
          provide a dataloader (these are listed in nncf.torch.dynamic_graph.io_handling.EXTRA_STRUCTS_WITH_DATALOADERS)

        :return: An initialized LoaderInputInfo object.
        """
        extra_structs = config.get_all_extra_structs()
        for extra_struct in extra_structs:
            if isinstance(extra_struct, tuple(EXTRA_STRUCTS_WITH_DATALOADERS)):
                extra_struct: HasDataloader
                dataloader = extra_struct.data_loader
                wrapped_dataloader = wrap_dataloader_for_init(dataloader)
                dataloader_output = next(iter(wrapped_dataloader))
                args, kwargs = wrapped_dataloader.get_inputs(dataloader_output)
                return LoaderInputInfo(args, kwargs)
        # config extra structs had no suitable dataloaders
        return None


class InputInfoWrapManager:
    def __init__(
        self, input_info: ModelInputInfo, fwd_signature: Signature, module_ref_for_device: torch.nn.Module = None
    ):
        self._fwd_signature = fwd_signature
        self._module_ref_for_device = module_ref_for_device
        args, kwargs = input_info.get_forward_inputs()
        bound_params = fwd_signature.bind(*args, **kwargs)
        self._fwd_param_names_to_dummy_inputs_odict: Dict[str, Any] = bound_params.arguments

    def wrap_inputs(self, model_args: Tuple, model_kwargs: Dict) -> Tuple[Tuple, Dict]:
        bound_model_params = self._fwd_signature.bind(*model_args, **model_kwargs)
        for param_name, dummy_input in self._fwd_param_names_to_dummy_inputs_odict.items():
            if not isinstance(dummy_input, torch.Tensor):
                continue  # Did not expect a tensor at this parameter during graph building, shouldn't wrap now too

            param_kind = self._fwd_signature.parameters[param_name].kind
            if param_kind is Parameter.VAR_POSITIONAL or param_kind is Parameter.VAR_KEYWORD:
                nncf_logger.warning(
                    "An input_info tensor was bound to a *args or **kwargs variadic parameter in the"
                    "forward's signature! This is currently unsupported by NNCF. Input compression may "
                    "be incorrect."
                )
                # Currently won't support input info mapping to *args or **kwargs-mapped parameters
                continue

            if param_name not in bound_model_params.arguments:
                nncf_logger.warning(
                    "A call to a compressed model's forward occurred without one of the arguments "
                    "specified at the compressed model creation stage! Input compression may be incorrect. "
                    "Trying to recover by wrapping the default value for the argument."
                )
                bound_model_params.apply_defaults()

            potential_tensor = bound_model_params.arguments[param_name]

            if potential_tensor is None:
                # Default was None - cannot wrap as-is. Will wrap a dummy tensor as specified in
                # input info - will preserve the call order of nncf_model_input nodes,
                # and the post-hooks for the input node will execute. The result won't go anywhere, though.
                nncf_logger.info(f"Wrapping a dummy tensor for input {param_name}")
                device = "cuda"
                if self._module_ref_for_device is not None:
                    device = get_model_device(self._module_ref_for_device)
                dummy_input_copy = dummy_input.clone().to(device)
                _ = nncf_model_input(dummy_input_copy)
            elif isinstance(potential_tensor, torch.Tensor):
                # Skip wrapping by nncf_model_input in case potential tensor is not a torch.Tensor.
                bound_model_params.arguments[param_name] = nncf_model_input(potential_tensor)

        return bound_model_params.args, bound_model_params.kwargs


class HasDataloader(Protocol):
    @property
    def data_loader(self) -> NNCFDataLoader:
        pass


EXTRA_STRUCTS_WITH_DATALOADERS: List[Type[HasDataloader]] = [QuantizationRangeInitArgs, BNAdaptationInitArgs]
