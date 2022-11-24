"""
 Copyright (c) 2022 Intel Corporation
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
from typing import List

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from nncf.torch.exporter import PTExporter


class BNTrainingStateSwitcher:
    """
    Context manager for switching between evaluation and training mode of BatchNormalization module.
    At the enter, it sets a forward pre-hook for setting BatchNormalization layers to the given state whether training
    or evaluation.
    At the exit, restore original BatchNormalization layer mode.
    """

    def __init__(self, model: nn.Module, is_training: bool = True):
        self.original_training_state = {}
        self.model = model
        self.is_training = is_training
        self.handles: List[RemovableHandle] = []

    @staticmethod
    def _apply_to_batchnorms(func):
        def func_apply_to_bns(module):
            if isinstance(module, (torch.nn.modules.batchnorm.BatchNorm1d,
                                   torch.nn.modules.batchnorm.BatchNorm2d,
                                   torch.nn.modules.batchnorm.BatchNorm3d)):
                func(module)

        return func_apply_to_bns

    def __enter__(self):
        def save_original_bn_training_state(module: torch.nn.Module):
            self.original_training_state[module] = module.training

        self.model.apply(self._apply_to_batchnorms(save_original_bn_training_state))

        def hook(module, _) -> None:
            module.training = self.is_training

        def register_hook(module: torch.nn.Module):
            handle = module.register_forward_pre_hook(hook)
            self.handles.append(handle)

        self.model.apply(self._apply_to_batchnorms(register_hook))
        return self

    def __exit__(self, *args):
        def restore_original_bn_training_state(module: torch.nn.Module):
            module.training = self.original_training_state[module]

        self.model.apply(self._apply_to_batchnorms(restore_original_bn_training_state))
        for handle in self.handles:
            handle.remove()


class NASExporter(PTExporter):
    """
    This class provides export of the NAS model to the ONNX format. The ordinary compressed models are exported in
    torch.onnx.TrainingMode.EVAL mode. This way leads to a hang for NAS model with elastic depth enabled.
    That's why NAS model is exported in torch.onnx.TrainingMode.TRAINING mode.
    """
    def _torch_export_call(self, model, input_tensor_list, save_path, input_names, output_names, opset_version):
        """
        Call of torch.onnx.export function.
        @param model: torch.nn.Module to be exported.
        @param input_tensor_list: the list containing model inputs.
        @param save_path: a string containing a path for saving onnx model.
        @param opset_version: the version of the onnx opset.
        @param output_names: Names to be assigned to the output tensors of the model.
        @param input_names: Names to be assigned to the input tensors of the model.
        """
        with BNTrainingStateSwitcher(model, is_training=False):
            torch.onnx.export(
                model, tuple(input_tensor_list), save_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                training=torch.onnx.TrainingMode.TRAINING
            )
