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

import atheris

from tests.cross_fw.shared.paths import TEST_ROOT

with atheris.instrument_imports():
    import nncf
    import sys

from random import randint
from random import seed

import openvino.runtime as ov
import torch
from torchvision import datasets
from torchvision import transforms

EXPECTED_LEN = 10


def create_mutated_name_list(name_list, data):
    if not name_list:
        return []
    seed(1)
    new_name_list = []

    for i in range(10):
        idx = randint(0, len(name_list) - 1)
        new_name_list.append(name_list[idx])

    mutated_id = data[0] % 10

    new_name_list[mutated_id] = f"_{new_name_list[mutated_id]}_"

    return new_name_list


def execute(model, in_dataset_path, f_mode, f_preset, f_target_device, f_subset_size, f_fast_bias_correction, data):
    if len(data) >= EXPECTED_LEN:
        dataset = datasets.ImageFolder(in_dataset_path, transforms.Compose([transforms.ToTensor()]))
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # Step 1: Initialize transformation function
        def transform_fn(data_item):
            images, _ = data_item
            return images

        # Step 2: Initialize NNCF Dataset
        calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
        # Step 3: Run the quantization pipeline

        ops = model.get_ordered_ops()
        name_list = [op.get_name() for op in ops]

        mutated_name_list = create_mutated_name_list(name_list, data)
        _ = nncf.IgnoredScope(names=[mutated_name_list])

        nncf.quantize(
            model=model,
            calibration_dataset=calibration_dataset,
            mode=f_mode,
            preset=f_preset,
            target_device=f_target_device,
            subset_size=f_subset_size,
            fast_bias_correction=f_fast_bias_correction,
            model_type=nncf.ModelType.TRANSFORMER,
            ignored_scope=None,
            advanced_parameters=None,
        )


def TestOneInput(data):
    in_model = ov.Model([], [])
    in_dataset_path = TEST_ROOT / "torch" / "data" / "mock_datasets" / "camvid"

    mode_list = [nncf.QuantizationMode.FP8_E4M3, nncf.QuantizationMode.FP8_E5M2]
    preset_list = [nncf.QuantizationPreset.PERFORMANCE, nncf.QuantizationPreset.MIXED]
    target_device_list = [
        nncf.TargetDevice.ANY,
        nncf.TargetDevice.CPU,
        nncf.TargetDevice.GPU,
        nncf.TargetDevice.NPU,
        nncf.TargetDevice.CPU_SPR,
    ]
    subset_size_list = [10, 11, 17]
    fast_bias_correction_list = [True, False]

    for f_mode in mode_list:
        for f_preset in preset_list:
            for f_target_device in target_device_list:
                for f_subset_size in subset_size_list:
                    for f_fast_bias_correction in fast_bias_correction_list:
                        execute(
                            in_model,
                            in_dataset_path,
                            f_mode,
                            f_preset,
                            f_target_device,
                            f_subset_size,
                            f_fast_bias_correction,
                            data,
                        )


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
