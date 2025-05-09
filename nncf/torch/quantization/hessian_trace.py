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
from typing import Any, Callable, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nncf.common.logging import nncf_logger
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_tensor


class ParameterHandler:
    def __init__(self, parameters: list[Parameter], device: str):
        self._device = device
        self._parameters = parameters

    @property
    def parameters(self) -> list[Parameter]:
        return self._parameters

    def get_gradients(self) -> list[Union[Tensor, float]]:
        gradients = []
        for parameter in self.parameters:
            gradients.append(0.0 if parameter.grad is None else parameter.grad + 0.0)
        return gradients

    def sample_rademacher_like_params(self) -> list[Tensor]:
        def sample(parameter):
            r = torch.randint_like(parameter, high=2, device=self._device)
            return r.masked_fill_(r == 0, -1)

        return [sample(p) for p in self.parameters]

    def sample_normal_like_params(self) -> list[Tensor]:
        return [torch.randn(p.size(), device=self._device) for p in self.parameters]


class GradientsCalculator:
    def __init__(
        self,
        model: nn.Module,
        criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor],
        criterion: _Loss,
        data_loader: PTInitializingDataLoader,
        num_data_iter: int,
        parameter_handler: ParameterHandler,
    ):
        self._model = model
        self._criterion_fn = criterion_fn
        self._criterion = criterion
        self._data_loader = data_loader
        self._num_data_iter = num_data_iter
        self._parameter_handler = parameter_handler
        self.num_iter = 0

    def __iter__(self):
        self.data_loader_iter = iter(self._data_loader)
        self.num_iter = 0
        return self

    def __next__(self):
        if self.num_iter >= self._num_data_iter:
            raise StopIteration
        self.num_iter += 1
        dataloader_output = next(self.data_loader_iter)

        device = get_model_device(self._model)
        to_device_fn = partial(torch.Tensor.to, device=device)
        dataloader_output = objwalk(dataloader_output, is_tensor, to_device_fn)
        args, kwargs = self._data_loader.get_inputs(dataloader_output)

        self._model.zero_grad()

        target = self._data_loader.get_target(dataloader_output)
        outputs = self._model(*args, **kwargs)
        loss = self._criterion_fn(outputs, target, self._criterion)

        loss.backward(create_graph=True)
        grads = self._parameter_handler.get_gradients()
        self._model.zero_grad()
        return grads


class HessianTraceEstimator:
    """
    Performs estimation of Hessian Trace based on Hutchinson algorithm.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor],
        criterion: _Loss,
        device: str,
        data_loader: DataLoader,
        num_data_points: int,
    ):
        self._model = model
        parameters = [p for p in model.parameters() if p.requires_grad]
        self._parameter_handler = ParameterHandler(parameters, device)
        self._batch_size = data_loader.batch_size
        data_loader = wrap_dataloader_for_init(data_loader)
        self._num_data_iter = num_data_points // self._batch_size if num_data_points >= self._batch_size else 1
        self._gradients_calculator = GradientsCalculator(
            self._model, criterion_fn, criterion, data_loader, self._num_data_iter, self._parameter_handler
        )
        self._diff_eps = 1e-6

    def get_average_traces(self, max_iter=500, tolerance=1e-5) -> Tensor:
        """
        Estimates average hessian trace for each parameter
        :param max_iter: maximum number of iterations for Hutchinson algorithm
        :param tolerance: - minimum relative tolerance for stopping the algorithm.
        It's calculated  between mean average trace from previous iteration and current one.
        :return: Tensor with average hessian trace per parameter
        """
        avg_total_trace = 0.0
        avg_traces_per_iter: list[Tensor] = []
        mean_avg_traces_per_param = None

        for i in range(max_iter):
            avg_traces_per_iter.append(self._calc_avg_traces_per_param())

            mean_avg_traces_per_param = self._get_mean(avg_traces_per_iter)
            mean_avg_total_trace = torch.sum(mean_avg_traces_per_param)

            diff_avg = abs(mean_avg_total_trace - avg_total_trace) / (abs(avg_total_trace) + self._diff_eps)
            if diff_avg < tolerance:
                return mean_avg_traces_per_param
            avg_total_trace = mean_avg_total_trace
            nncf_logger.debug(f"{i}# difference_avg={diff_avg} avg_trace={avg_total_trace}")

        return mean_avg_traces_per_param

    def _calc_avg_traces_per_param(self) -> Tensor:
        v = self._parameter_handler.sample_rademacher_like_params()
        vhp = self._parameter_handler.sample_normal_like_params()
        num_all_data = self._num_data_iter * self._batch_size
        for gradients in self._gradients_calculator:
            vhp_curr = torch.autograd.grad(
                gradients, self._parameter_handler.parameters, grad_outputs=v, only_inputs=True, retain_graph=False
            )
            vhp = [a + b * float(self._batch_size) + 0.0 for a, b in zip(vhp, vhp_curr)]
        vhp = [a / float(num_all_data) for a in vhp]
        avg_traces_per_param = torch.stack([torch.sum(a * b) / a.size().numel() for (a, b) in zip(vhp, v)])
        return avg_traces_per_param

    @staticmethod
    def _get_mean(data: list[Tensor]) -> Tensor:
        return torch.mean(torch.stack(data), dim=0)
