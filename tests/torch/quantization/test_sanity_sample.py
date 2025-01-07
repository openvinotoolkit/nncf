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

from abc import ABC
from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import nn

from nncf import NNCFConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.torch.quantization.algo import QuantizationController
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.sample_test_validator import SampleType
from tests.torch.sample_test_validator import SanitySampleValidator
from tests.torch.sample_test_validator import SanityTestCaseDescriptor


class PrecisionTestCaseDescriptor(SanityTestCaseDescriptor, ABC):
    def __init__(self):
        super().__init__()
        self.num_weight_quantizers_: int = 0
        self.num_activation_quantizers_: int = 0

    @property
    def config_directory(self) -> Path:
        return TEST_ROOT / "torch" / "data" / "configs" / "hawq"

    def get_precision_section(self) -> Dict:
        raise NotImplementedError

    def get_compression_section(self):
        quantization_section = {
            "algorithm": "quantization",
            "initializer": {
                "precision": self.get_precision_section(),
                "range": {"num_init_samples": 2},
                "batchnorm_adaptation": {"num_bn_adaptation_samples": 1},
            },
        }
        if self.sample_type_ == SampleType.CLASSIFICATION_STAGED:
            quantization_section.update({"params": {"activations_quant_start_epoch": 0}})
        return quantization_section

    def num_weight_quantizers(self, n: int):
        self.num_weight_quantizers_ = n
        return self

    def num_activation_quantizers(self, n: int):
        self.num_activation_quantizers_ = n
        return self


class PrecisionSampleValidator(SanitySampleValidator):
    def __init__(self, desc: PrecisionTestCaseDescriptor):
        super().__init__(desc)
        self._train_mock = None

    def setup_spy(self, mocker):
        train_location = self._sample_handler.get_train_location()
        self._train_mock = mocker.patch(train_location)

    def validate_spy(self):
        self._train_mock.assert_called_once()


class HAWQTestCaseDescriptor(PrecisionTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.batch_size_init_: int = 0

    def get_validator(self):
        return HAWQSampleValidator(self)

    def batch_size_init(self, batch_size_init: int):
        self.batch_size_init_ = batch_size_init
        return self

    def get_sample_params(self):
        result = super().get_sample_params()
        result.update({"batch_size_init": self.batch_size_init_} if self.batch_size_init_ else {})
        return result

    def get_precision_section(self) -> Dict:
        return {"type": "hawq", "num_data_points": 3, "iter_number": 1}

    def __str__(self):
        bs = f"_bs{self.batch_size_init_}" if self.batch_size_init_ else ""
        return super().__str__() + "_hawq" + bs


class HAWQSampleValidator(PrecisionSampleValidator):
    def __init__(self, desc: HAWQTestCaseDescriptor):
        super().__init__(desc)
        self._desc = desc
        self.get_qsetup_spy = None
        self.hessian_trace_estimator_spy = None

    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        from nncf.torch.quantization.init_precision import HAWQPrecisionInitializer

        self.get_qsetup_spy = mocker.spy(HAWQPrecisionInitializer, "get_quantizer_setup_for_qconfig_sequence")
        from nncf.torch.quantization.hessian_trace import HessianTraceEstimator

        self.hessian_trace_estimator_spy = mocker.spy(HessianTraceEstimator, "__init__")

    def validate_spy(self):
        super().validate_spy()
        qconfig_sequence = self.get_qsetup_spy.call_args[0][1]
        assert len(qconfig_sequence) == self._desc.num_weight_quantizers_
        all_precisions = {qc.num_bits for qc in qconfig_sequence}
        # with default compression ratio = 1.5 all precisions should be different from the default one
        assert all_precisions != {QuantizerConfig().num_bits}

        init_data_loader = self.hessian_trace_estimator_spy.call_args[0][5]
        expected_batch_size = self._desc.batch_size_init_ if self._desc.batch_size_init_ else self._desc.batch_size_
        assert init_data_loader.batch_size == expected_batch_size


class AutoQTestCaseDescriptor(PrecisionTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.subset_ratio_: float = 1.0
        self.BITS = [2, 4, 8]
        self.debug_dump: bool = False

    def get_validator(self):
        return AutoQSampleValidator(self)

    def subset_ratio(self, subset_ratio_: float):
        self.subset_ratio_ = subset_ratio_
        return self

    def dump_debug(self, debug_dump: bool):
        self.debug_dump = debug_dump
        return self

    def get_precision_section(self) -> Dict:
        return {
            "type": "autoq",
            "bits": self.BITS,
            "iter_number": 2,
            "compression_ratio": 0.15,
            "eval_subset_ratio": self.subset_ratio_,
            "dump_init_precision_data": self.debug_dump,
        }

    def __str__(self):
        sr = f"_sr{self.subset_ratio_}" if self.subset_ratio_ else ""
        dd = "_dump_debug" if self.debug_dump else ""
        return super().__str__() + "_autoq" + sr + dd


class AutoQSampleValidator(PrecisionSampleValidator):
    def __init__(self, desc: AutoQTestCaseDescriptor):
        super().__init__(desc)
        self._desc = desc
        self.builder_spy = None

    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        from nncf.torch.quantization.algo import QuantizationBuilder

        self.builder_spy = mocker.spy(QuantizationBuilder, "build_controller")

    def validate_spy(self):
        super().validate_spy()
        ctrl = self.builder_spy.spy_return
        final_bits = [qm.num_bits for qm in ctrl.all_quantizations.values()]
        assert set(final_bits) != {QuantizerConfig().num_bits}
        assert all(bit in self._desc.BITS for bit in final_bits)


def resnet18_desc(x: PrecisionTestCaseDescriptor):
    return (
        x.config_name("resnet18_cifar10_mixed_int.json")
        .sample_type(SampleType.CLASSIFICATION)
        .mock_dataset("mock_32x32")
        .batch_size(3)
        .num_weight_quantizers(21)
        .num_activation_quantizers(27)
    )


def inception_v3_desc(x: PrecisionTestCaseDescriptor):
    return (
        x.config_name("inception_v3_cifar10_mixed_int.json")
        .sample_type(SampleType.CLASSIFICATION)
        .mock_dataset("mock_32x32")
        .batch_size(3)
        .num_weight_quantizers(95)
        .num_activation_quantizers(105)
    )


def ssd300_vgg_desc(x: PrecisionTestCaseDescriptor):
    return (
        x.config_name("ssd300_vgg_voc_mixed_int.json")
        .sample_type(SampleType.OBJECT_DETECTION)
        .mock_dataset("voc")
        .batch_size(3)
        .num_weight_quantizers(35)
        .num_activation_quantizers(27)
    )


def unet_desc(x: PrecisionTestCaseDescriptor):
    return (
        x.config_name("unet_camvid_mixed_int.json")
        .sample_type(SampleType.SEMANTIC_SEGMENTATION)
        .mock_dataset("camvid")
        .batch_size(3)
        .num_weight_quantizers(23)
        .num_activation_quantizers(23)
    )


def icnet_desc(x: PrecisionTestCaseDescriptor):
    return (
        x.config_name("icnet_camvid_mixed_int.json")
        .sample_type(SampleType.SEMANTIC_SEGMENTATION)
        .mock_dataset("camvid")
        .batch_size(3)
        .num_weight_quantizers(64)
        .num_activation_quantizers(81)
    )


TEST_CASE_DESCRIPTORS = [
    inception_v3_desc(HAWQTestCaseDescriptor()),
    inception_v3_desc(HAWQTestCaseDescriptor()).sample_type(SampleType.CLASSIFICATION_STAGED),
    resnet18_desc(HAWQTestCaseDescriptor()),
    resnet18_desc(HAWQTestCaseDescriptor()).sample_type(SampleType.CLASSIFICATION_STAGED),
    resnet18_desc(HAWQTestCaseDescriptor().batch_size_init(2)),
    resnet18_desc(HAWQTestCaseDescriptor().batch_size_init(2)).sample_type(SampleType.CLASSIFICATION_STAGED),
    ssd300_vgg_desc(HAWQTestCaseDescriptor()),
    ssd300_vgg_desc(HAWQTestCaseDescriptor().batch_size_init(2)),
    unet_desc(HAWQTestCaseDescriptor()),
    unet_desc(HAWQTestCaseDescriptor().batch_size_init(2)),
    icnet_desc(HAWQTestCaseDescriptor()),
    inception_v3_desc(AutoQTestCaseDescriptor()).batch_size(2),
    inception_v3_desc(AutoQTestCaseDescriptor()).sample_type(SampleType.CLASSIFICATION_STAGED),
    resnet18_desc(AutoQTestCaseDescriptor()).batch_size(2),
    resnet18_desc(AutoQTestCaseDescriptor().dump_debug(True))
    .batch_size(2)
    .sample_type(SampleType.CLASSIFICATION_STAGED),
    resnet18_desc(AutoQTestCaseDescriptor().subset_ratio(0.2)).batch_size(2),
    resnet18_desc(AutoQTestCaseDescriptor().subset_ratio(0.2)).sample_type(SampleType.CLASSIFICATION_STAGED),
    ssd300_vgg_desc(AutoQTestCaseDescriptor().dump_debug(True)).batch_size(2),
    unet_desc(AutoQTestCaseDescriptor().dump_debug(True)),
    icnet_desc(AutoQTestCaseDescriptor()),
]


@pytest.fixture(name="precision_desc", params=TEST_CASE_DESCRIPTORS, ids=map(str, TEST_CASE_DESCRIPTORS))
def fixture_precision_desc(request, dataset_dir):
    desc: PrecisionTestCaseDescriptor = request.param
    return desc.finalize(dataset_dir)


@pytest.mark.nightly
def test_precision_init(precision_desc: PrecisionTestCaseDescriptor, tmp_path, mocker):
    validator = precision_desc.get_validator()
    args = validator.get_default_args(tmp_path)
    validator.validate_sample(args, mocker)


class ExportTestCaseDescriptor(PrecisionTestCaseDescriptor):
    def get_validator(self):
        return ExportSampleValidator(self)

    def get_precision_section(self) -> Dict:
        return {}

    def get_sample_params(self):
        result = super().get_sample_params()
        result.update({"pretrained": True})
        return result


class ExportSampleValidator(PrecisionSampleValidator):
    def __init__(self, desc: ExportTestCaseDescriptor):
        super().__init__(desc)
        self._desc = desc
        self.is_export_called = False
        self._ctrl_mock = None
        self._reg_init_args_patch = None
        self._create_compressed_model_patch = None

    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        self._reg_init_args_patch = mocker.spy(NNCFConfig, "register_extra_structs")
        sample_location = self._sample_handler.get_sample_location()

        if self._desc.sample_type_ == SampleType.OBJECT_DETECTION:
            mocker.patch(sample_location + ".build_ssd")
        else:
            load_model_location = sample_location + ".load_model"
            mocker.patch(load_model_location)

        ctrl_mock = mocker.MagicMock(spec=QuantizationController)
        model_mock = mocker.MagicMock(spec=nn.Module)
        mocker.patch("examples.torch.common.export.get_export_args", return_value=((torch.Tensor([1, 1]),), {}))
        create_model_location = sample_location + ".create_compressed_model"
        create_model_patch = mocker.patch(create_model_location)

        self._torch_onn_export_mock = mocker.patch("torch.onnx.export")

        if self._desc.sample_type_ == SampleType.CLASSIFICATION_STAGED:
            mocker.patch(sample_location + ".get_quantization_optimizer")

        def fn(*args, **kwargs):
            return ctrl_mock, model_mock

        create_model_patch.side_effect = fn
        self._ctrl_mock = ctrl_mock

    def validate_spy(self):
        super().validate_spy()
        self._reg_init_args_patch.assert_called()

        if self.is_export_called:
            self._torch_onn_export_mock.assert_called_once()
        else:
            self._torch_onn_export_mock.assert_not_called()


EXPORT_TEST_CASE_DESCRIPTORS = [
    resnet18_desc(ExportTestCaseDescriptor()),
    resnet18_desc(ExportTestCaseDescriptor()).sample_type(SampleType.CLASSIFICATION_STAGED),
    ssd300_vgg_desc(ExportTestCaseDescriptor()),
    unet_desc(ExportTestCaseDescriptor()),
]


@pytest.fixture(name="export_desc", params=EXPORT_TEST_CASE_DESCRIPTORS, ids=map(str, EXPORT_TEST_CASE_DESCRIPTORS))
def fixture_export_desc(request):
    desc: PrecisionTestCaseDescriptor = request.param
    return desc.finalize()


@pytest.mark.nightly
@pytest.mark.parametrize(
    ("extra_args", "is_export_called"),
    (({}, False), ({"-m": ["export", "train"]}, True)),
    ids=["train_with_onnx_path", "export_after_train"],
)
def test_export_behavior(export_desc: PrecisionTestCaseDescriptor, tmp_path, mocker, extra_args, is_export_called):
    validator = export_desc.get_validator()
    args = validator.get_default_args(tmp_path)
    args["--export-model-path"] = tmp_path / "model.onnx"
    if extra_args is not None:
        args.update(extra_args)
    validator.is_export_called = is_export_called
    validator.validate_sample(args, mocker)
