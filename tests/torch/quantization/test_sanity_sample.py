from typing import Dict

import pytest
from torch import nn

from nncf import NNCFConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.torch.quantization.algo import QuantizationController
from tests.torch.test_sanity_sample import SampleType
from tests.torch.test_sanity_sample import SanityTestCaseDescriptor
from tests.torch.test_sanity_sample import get_default_args
from tests.torch.test_sanity_sample import validate_sample


class PrecisionTestCaseDescriptor(SanityTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.n_weight_quantizers: int = 0
        self.n_activation_quantizers: int = 0
        self.quantization_algo_params: Dict = {}
        self._train_mock = None

    def get_precision_section(self) -> Dict:
        raise NotImplementedError

    def get_compression_section(self):
        return {
            'algorithm': 'quantization',
            'initializer': {
                'precision': self.get_precision_section(),
                'range': {
                    "num_init_samples": 2
                },
                "batchnorm_adaptation": {
                    "num_bn_adaptation_samples": 1
                }
            },
            'params': self.quantization_algo_params,
        }

    def staged(self):
        super().staged()
        self.quantization_algo_params = {
            "activations_quant_start_epoch": 0
        }
        return self

    def setup_spy(self, mocker):
        train_location = self.get_train_location()
        self._train_mock = mocker.patch(train_location)

        # Need to mock SafeMLFLow to prevent starting a not closed mlflow session due to memory leak of config and
        # SafeMLFLow, which happens with a mocked train function
        mlflow_location = self.get_sample_file_location() + '.SafeMLFLow'
        mocker.patch(mlflow_location)

    def validate_spy(self):
        self._train_mock.assert_called_once()

    def num_weight_quantizers(self, n: int):
        self.n_weight_quantizers = n
        return self

    def num_activation_quantizers(self, n: int):
        self.n_activation_quantizers = n
        return self


class HAWQTestCaseDescriptor(PrecisionTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.batch_size_init: int = 0
        self.get_qsetup_spy = None
        self.hessian_trace_estimator_spy = None

    def batch_for_init(self, batch_size_init: int):
        self.batch_size_init = batch_size_init
        return self

    def get_sample_params(self):
        result = super().get_sample_params()
        result.update({'batch_size_init': self.batch_size_init} if self.batch_size_init else {})
        return result

    def get_precision_section(self) -> Dict:
        return {"type": "hawq",
                "num_data_points": 3,
                "iter_number": 1}

    def __str__(self):
        bs = f'_bs{self.batch_size_init}' if self.batch_size_init else ''
        return super().__str__() + '_hawq' + bs

    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        from nncf.torch.quantization.init_precision import HAWQPrecisionInitializer
        self.get_qsetup_spy = mocker.spy(HAWQPrecisionInitializer, "get_quantizer_setup_for_qconfig_sequence")
        from nncf.torch.quantization.hessian_trace import HessianTraceEstimator
        self.hessian_trace_estimator_spy = mocker.spy(HessianTraceEstimator, "__init__")

    def validate_spy(self):
        super().validate_spy()
        qconfig_sequence = self.get_qsetup_spy.call_args[0][1]
        assert len(qconfig_sequence) == self.n_weight_quantizers
        all_precisions = {qc.num_bits for qc in qconfig_sequence}
        # with default compression ratio = 1.5 all precisions should be different from the default one
        assert all_precisions != {QuantizerConfig().num_bits}

        init_data_loader = self.hessian_trace_estimator_spy.call_args[0][5]
        expected_batch_size = self.batch_size_init if self.batch_size_init else self.batch_size
        assert init_data_loader.batch_size == expected_batch_size


class AutoQTestCaseDescriptor(PrecisionTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.subset_ratio_: float = 1.0
        self.BITS = [2, 4, 8]
        self.debug_dump: bool = False

    def subset_ratio(self, subset_ratio_: float):
        self.subset_ratio_ = subset_ratio_
        return self

    def dump_debug(self, debug_dump: bool):
        self.debug_dump = debug_dump
        return self

    def get_precision_section(self) -> Dict:
        return {"type": "autoq",
                "bits": self.BITS,
                "iter_number": 2,
                "compression_ratio": 0.15,
                "eval_subset_ratio": self.subset_ratio_,
                "dump_init_precision_data": self.debug_dump}

    def __str__(self):
        sr = f'_sr{self.subset_ratio_}' if self.subset_ratio_ else ''
        dd = '_dump_debug' if self.debug_dump else ''
        return super().__str__() + '_autoq' + sr + dd

    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        from nncf.torch.quantization.algo import QuantizationBuilder
        self.builder_spy = mocker.spy(QuantizationBuilder, 'build_controller')

    def validate_spy(self):
        super().validate_spy()
        ctrl = self.builder_spy.spy_return
        final_bits = [qm.num_bits for qm in ctrl.all_quantizations.values()]
        assert set(final_bits) != {QuantizerConfig().num_bits}
        assert all(bit in self.BITS for bit in final_bits)


def resnet18_desc(x: PrecisionTestCaseDescriptor):
    return x.hawq_config("resnet18_cifar10_mixed_int.json").sample(SampleType.CLASSIFICATION). \
        mock_dataset('mock_32x32').batch(3).num_weight_quantizers(21).num_activation_quantizers(27)


def inception_v3_desc(x: PrecisionTestCaseDescriptor):
    return x.hawq_config("inception_v3_cifar10_mixed_int.json").sample(SampleType.CLASSIFICATION). \
        mock_dataset('mock_32x32').batch(3).num_weight_quantizers(95).num_activation_quantizers(105)


def ssd300_vgg_desc(x: PrecisionTestCaseDescriptor):
    return x.hawq_config("ssd300_vgg_voc_mixed_int.json").sample(SampleType.OBJECT_DETECTION). \
        mock_dataset('voc').batch(3).num_weight_quantizers(35).num_activation_quantizers(27)


def unet_desc(x: PrecisionTestCaseDescriptor):
    return x.hawq_config("unet_camvid_mixed_int.json").sample(SampleType.SEMANTIC_SEGMENTATION). \
        mock_dataset('camvid').batch(3).num_weight_quantizers(23).num_activation_quantizers(23)


def icnet_desc(x: PrecisionTestCaseDescriptor):
    return x.hawq_config("icnet_camvid_mixed_int.json").sample(SampleType.SEMANTIC_SEGMENTATION). \
        mock_dataset('camvid').batch(3).num_weight_quantizers(64).num_activation_quantizers(81)


TEST_CASE_DESCRIPTORS = [
    inception_v3_desc(HAWQTestCaseDescriptor()),
    inception_v3_desc(HAWQTestCaseDescriptor()).staged(),
    resnet18_desc(HAWQTestCaseDescriptor()),
    resnet18_desc(HAWQTestCaseDescriptor()).staged(),
    resnet18_desc(HAWQTestCaseDescriptor().batch_for_init(2)),
    resnet18_desc(HAWQTestCaseDescriptor().batch_for_init(2)).staged(),
    ssd300_vgg_desc(HAWQTestCaseDescriptor()),
    ssd300_vgg_desc(HAWQTestCaseDescriptor().batch_for_init(2)),
    unet_desc(HAWQTestCaseDescriptor()),
    unet_desc(HAWQTestCaseDescriptor().batch_for_init(2)),
    icnet_desc(HAWQTestCaseDescriptor()),
    inception_v3_desc(AutoQTestCaseDescriptor()).batch(2),
    inception_v3_desc(AutoQTestCaseDescriptor()).staged(),
    resnet18_desc(AutoQTestCaseDescriptor()).batch(2),
    resnet18_desc(AutoQTestCaseDescriptor().dump_debug(True)).batch(2).staged(),
    resnet18_desc(AutoQTestCaseDescriptor().subset_ratio(0.2)).batch(2),
    resnet18_desc(AutoQTestCaseDescriptor().subset_ratio(0.2)).staged(),
    ssd300_vgg_desc(AutoQTestCaseDescriptor().dump_debug(True)).batch(2),
    unet_desc(AutoQTestCaseDescriptor().dump_debug(True)),
    icnet_desc(AutoQTestCaseDescriptor())
]


@pytest.fixture(name='precision_desc', params=TEST_CASE_DESCRIPTORS, ids=map(str, TEST_CASE_DESCRIPTORS))
def fixture_precision_desc(request, dataset_dir):
    desc: PrecisionTestCaseDescriptor = request.param
    return desc.finalize(dataset_dir)


def test_precision_init(precision_desc: PrecisionTestCaseDescriptor, tmp_path, mocker):
    args = get_default_args(precision_desc, tmp_path)
    validate_sample(args, precision_desc, mocker)


class ExportTestCaseDescriptor(PrecisionTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self._create_compressed_model_patch = None
        self._reg_init_args_patch = None
        self._ctrl_mock = None
        self.is_export_called = False

    def get_precision_section(self) -> Dict:
        return {}

    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        self._reg_init_args_patch = mocker.spy(NNCFConfig, "register_extra_structs")
        sample_file_location = self.get_sample_file_location()

        if self.sample_type == SampleType.OBJECT_DETECTION:
            mocker.patch(sample_file_location + '.build_ssd')
        else:
            load_model_location = sample_file_location + '.load_model'
            mocker.patch(load_model_location)

        ctrl_mock = mocker.MagicMock(spec=QuantizationController)
        model_mock = mocker.MagicMock(spec=nn.Module)
        create_model_location = sample_file_location + '.create_compressed_model'
        create_model_patch = mocker.patch(create_model_location)

        if self.is_staged:
            mocker.patch(sample_file_location + '.get_quantization_optimizer')

        def fn(*args, **kwargs):
            return ctrl_mock, model_mock

        create_model_patch.side_effect = fn
        self._ctrl_mock = ctrl_mock

    def validate_spy(self):
        super().validate_spy()
        self._reg_init_args_patch.assert_called()
        if self.is_export_called:
            self._ctrl_mock.export_model.assert_called_once()
        else:
            self._ctrl_mock.export_model.assert_not_called()

    def get_sample_params(self):
        result = super().get_sample_params()
        result.update({'pretrained': True})
        return result


EXPORT_TEST_CASE_DESCRIPTORS = [
    resnet18_desc(ExportTestCaseDescriptor()),
    resnet18_desc(ExportTestCaseDescriptor()).staged(),
    ssd300_vgg_desc(ExportTestCaseDescriptor()),
    unet_desc(ExportTestCaseDescriptor()),
]


@pytest.fixture(name='export_desc', params=EXPORT_TEST_CASE_DESCRIPTORS, ids=map(str, EXPORT_TEST_CASE_DESCRIPTORS))
def fixture_export_desc(request):
    desc: PrecisionTestCaseDescriptor = request.param
    return desc.finalize()


@pytest.mark.parametrize(
    ('extra_args', 'is_export_called'),
    (
        ({}, False),
        ({"-m": 'export train'}, True)
    ),
    ids=['train_with_onnx_path', 'export_after_train']
)
def test_export_behavior(export_desc: PrecisionTestCaseDescriptor, tmp_path, mocker, extra_args, is_export_called):
    args = get_default_args(export_desc, tmp_path)
    args["--to-onnx"] = tmp_path / 'model.onnx'
    if extra_args is not None:
        args.update(extra_args)
    export_desc.is_export_called = is_export_called
    validate_sample(args, export_desc, mocker)
