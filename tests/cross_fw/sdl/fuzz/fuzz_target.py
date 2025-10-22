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

import contextlib
import os
import sys

import atheris
import numpy as np
import openvino as ov
from atheris import FuzzedDataProvider

with atheris.instrument_imports():
    import nncf

# Disable logging for cleaner fuzzing output
nncf.set_log_level(40)

# To disable telemetry during fuzzing
os.environ["NNCF_CI"] = "1"


class MockDataset:
    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < 300:
            self.n += 1
            return np.ones((1, 1, 1, 1), dtype=np.float32)
        raise StopIteration


def check_quantize_api(fdp: FuzzedDataProvider) -> None:
    model = ov.Model([], [])
    dataset = nncf.Dataset(MockDataset())
    r_mode = fdp.PickValueInList(list(nncf.QuantizationMode) + [None])
    r_preset = fdp.PickValueInList(list(nncf.QuantizationPreset) + [None])
    r_device = fdp.PickValueInList(list(nncf.TargetDevice))
    r_subset_size = fdp.ConsumeIntInRange(-10, 500)
    r_fbc = fdp.ConsumeBool()
    r_model_type = fdp.PickValueInList(list(nncf.ModelType) + [None])  # Keeping it None for simplicity

    with contextlib.suppress(nncf.ParameterNotSupportedError, nncf.ValidationError):
        nncf.quantize(
            model,
            calibration_dataset=dataset,
            mode=r_mode,
            preset=r_preset,
            target_device=r_device,
            subset_size=r_subset_size,
            fast_bias_correction=r_fbc,
            model_type=r_model_type,
        )


def check_compress_weights_api(fdp: FuzzedDataProvider) -> None:
    model = ov.Model([], [])
    dataset = nncf.Dataset(MockDataset())

    r_mode = fdp.PickValueInList(list(nncf.CompressWeightsMode))
    r_ratio = fdp.ConsumeFloatInRange(-0.1, 1.1) if fdp.ConsumeBool() else None
    r_group_size = fdp.ConsumeIntInRange(-10, 1000) if fdp.ConsumeBool() else None
    r_all_layers = fdp.PickValueInList([True, False, None])
    r_dataset = dataset if fdp.ConsumeBool() else None
    r_sensitivity_metric = fdp.PickValueInList(list(nncf.SensitivityMetric) + [None])
    r_subset_size = fdp.ConsumeIntInRange(-10, 400)
    r_awq = fdp.PickValueInList([True, False, None])
    r_scale_estimation = fdp.PickValueInList([True, False, None])
    r_gptq = fdp.PickValueInList([True, False, None])
    r_lora_correction = fdp.PickValueInList([True, False, None])
    r_backup_mode = fdp.PickValueInList(list(nncf.BackupMode) + [None])
    r_compression_format = fdp.PickValueInList(list(nncf.CompressionFormat))
    with contextlib.suppress(nncf.ParameterNotSupportedError, nncf.ValidationError):
        nncf.compress_weights(
            model,
            mode=r_mode,
            ratio=r_ratio,
            group_size=r_group_size,
            ignored_scope=None,
            dataset=r_dataset,
            sensitivity_metric=r_sensitivity_metric,
            all_layers=r_all_layers,
            subset_size=r_subset_size,
            awq=r_awq,
            scale_estimation=r_scale_estimation,
            gptq=r_gptq,
            lora_correction=r_lora_correction,
            backup_mode=r_backup_mode,
            compression_format=r_compression_format,
        )


def TestOneInput(data: bytes) -> None:
    fdp = FuzzedDataProvider(data)
    algo = fdp.PickValueInList(["ptq", "wc"])
    if algo == "ptq":
        check_quantize_api(fdp)
    elif algo == "wc":
        check_compress_weights_api(fdp)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
