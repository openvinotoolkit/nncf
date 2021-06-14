from collections import OrderedDict
from inspect import Signature, Parameter
from typing import Any
from typing import List

import torch

from nncf.common.graph import MODEL_INPUT_OP_NAME
from nncf.common.graph import MODEL_OUTPUT_OP_NAME
from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo, create_mock_tensor
from nncf.torch.utils import is_tensor, objwalk, is_traced_tensor
from nncf.common.utils.logger import logger as nncf_logger


@register_operator(name=MODEL_INPUT_OP_NAME)
def nncf_model_input(tensor: 'torch.Tensor'):
    return tensor

@register_operator(name=MODEL_OUTPUT_OP_NAME)
def nncf_model_output(tensor: 'torch.Tensor'):
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
    observed_tensor_object_ids = set()  # type: Set[int]

    def replicate_fn(tensor: torch.Tensor) -> torch.Tensor:
        tensor_object_id = id(tensor)
        if tensor_object_id in observed_tensor_object_ids:
            return tensor.clone()
        observed_tensor_object_ids.add(tensor_object_id)
        return tensor
    obj = objwalk(obj, is_tensor, replicate_fn)
    return obj


class InputInfoWrapManager:
    INPUTS_MISMATCH_WARNING_TEXT = "Compression with regards to this input may occur incorrectly. Make sure " \
                                   "you call the compressed model with inputs that correspond to what NNCF was " \
                                   "configured to expect (either via NNCF config's input_infos, or custom" \
                                   "dummy_forward_fn/wrap_inputs_fn parameters), or that you know what you are " \
                                   "doing. This warning will not be shown again."
    ARGS_INPUTS_MISMATCH_FORMAT_STRING = "Inputs mismatch - could not find arg with idx {} in NNCF-wrapped model " \
                                         "input args! " + INPUTS_MISMATCH_WARNING_TEXT
    KWARGS_INPUTS_MISMATCH_FORMAT_STRING = "Inputs mismatch - could not find kwarg '{}' in NNCF-wrapped model input " \
                                           "kwargs! " + INPUTS_MISMATCH_WARNING_TEXT

    def __init__(self, input_infos: List[ModelInputInfo],
                 fwd_signature: Signature,
                 module_ref_for_device: torch.nn.Module = None):
        self._module_ref_for_device = module_ref_for_device
        arg_iis_list = [ii for ii in input_infos if ii.keyword is None]
        kwarg_iis_list = [(ii.keyword, ii) for ii in input_infos if ii.keyword is not None]
        kwarg_iis = OrderedDict()
        arg_iis = tuple(arg_iis_list)
        for kw, ii in kwarg_iis_list:
            kwarg_iis[kw] = ii
        bound_params = fwd_signature.bind(*arg_iis, **kwarg_iis)

        self._fwd_params_to_input_infos_odict = bound_params.arguments
        self._fwd_signature = fwd_signature  # type: Signature

    def set_device(self, device: str):
        self._device = device

    def wrap_inputs(self, model_args, model_kwargs):
        bound_model_params = self._fwd_signature.bind(*model_args, **model_kwargs)
        for param_name in self._fwd_params_to_input_infos_odict:
            param_kind = self._fwd_signature.parameters[param_name].kind
            if param_kind is Parameter.VAR_POSITIONAL or param_kind is Parameter.VAR_KEYWORD:
                nncf_logger.warning("An input_info tensor was bound to a *args or **kwargs variadic parameter in the"
                                    "forward's signature! This is currently unsupported by NNCF. Input compression may "
                                    "be incorrect.")
                # Currently won't support input info mapping to *args or **kwargs-mapped parameters
                continue

            if param_name not in bound_model_params.arguments:
                nncf_logger.warning("A call to a compressed model's forward occured without one of the params"
                                    "specified in input_infos! Input compression may be incorrect. Trying to recover "
                                    "by wrapping the default value for the parameter.")
                bound_model_params.apply_defaults()

            potential_tensor = bound_model_params.arguments[param_name]
            if potential_tensor is not None:
                bound_model_params.arguments[param_name] = nncf_model_input(bound_model_params.arguments[param_name])
            else:
                # Default was None - cannot wrap as-is. Will wrap a dummy tensor as specified in
                # input infos - will conserve the call order of nncf_model_input nodes,
                # and the post-hooks for the input node will execute. The result won't go anywhere, though.
                nncf_logger.warning("Wrapping a dummy tensor for input {}".format(param_name))
                info_for_missing_input = self._fwd_params_to_input_infos_odict[param_name]
                device = 'cuda'
                if self._module_ref_for_device is not None:
                    device = next(self._module_ref_for_device.parameters()).device
                dummy_tensor = create_mock_tensor(info_for_missing_input, device)
                _ = nncf_model_input(dummy_tensor)

        return bound_model_params.args, bound_model_params.kwargs
