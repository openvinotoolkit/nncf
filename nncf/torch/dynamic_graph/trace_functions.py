"""
 Copyright (c) 2021 Intel Corporation
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

from copy import deepcopy
from typing import Callable

from torch import Tensor

from nncf.torch.dynamic_graph.trace_tensor import flatten_args
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor


class CustomTraceFunction:
    def __call__(self, operator: Callable, *args, **kwargs):
        raise NotImplementedError


class ForwardTraceOnly(CustomTraceFunction):
    def __call__(self, operator: Callable, *args, **kwargs):
        """
        This wrapper override will result in the operator not being added to graph,
        but the result will still have TracedTensors with parent IDs left the same as in input.
        Useful for operators which are not likely to be present in patterns considered for
        compression, but still have to be accounted for so that the NNCF internal graph representation
        does not become disjoint.
        """

        result = operator(*args, **kwargs)

        fargs = flatten_args(args, kwargs)
        input_traced_tensor_indices = [i for i in range(len(fargs)) if isinstance(fargs[i], TracedTensor)]

        if isinstance(result, (list, tuple)):
            output_tensors_to_be_traced_indices = [i for i in range(len(result)) if
                                                   isinstance(result[i], Tensor)]

            was_tuple = isinstance(result, tuple)
            result = list(result)
            if len(input_traced_tensor_indices) == 1:
                # Broadcast one and the same creator ID of input to all outputs
                for out_idx in output_tensors_to_be_traced_indices:
                    forwarded_meta = deepcopy(fargs[input_traced_tensor_indices[0]].tensor_meta)
                    forwarded_meta.shape = tuple(result[out_idx].shape)
                    result[out_idx] = TracedTensor.from_torch_tensor(result[out_idx],
                                                                     forwarded_meta)
            elif len(input_traced_tensor_indices) != len(output_tensors_to_be_traced_indices):
                raise RuntimeError("Unable to forward trace through operator {} - "
                                   "input and output tensor count mismatch!".format(operator.__name__))
            else:
                # Assume that output tensor order corresponds to input tensor order
                for in_idx, out_idx in zip(input_traced_tensor_indices, output_tensors_to_be_traced_indices):
                    forwarded_meta = deepcopy(fargs[in_idx].tensor_meta)
                    forwarded_meta.shape = tuple(result[out_idx].shape)
                    result[out_idx] = TracedTensor.from_torch_tensor(result[out_idx],
                                                                     forwarded_meta)
            if was_tuple:
                result = tuple(result)
        elif len(input_traced_tensor_indices) > 1:
            raise RuntimeError("Unable to forward trace through operator {} - "
                               "input and output tensor count mismatch!".format(operator.__name__))
        elif input_traced_tensor_indices:
            forwarded_meta = deepcopy(fargs[input_traced_tensor_indices[0]].tensor_meta)
            forwarded_meta.shape = tuple(result.shape)
            return TracedTensor.from_torch_tensor(result,
                                                  forwarded_meta)
        # No traced tensors in input, return a usual torch.Tensor as well
        return result
