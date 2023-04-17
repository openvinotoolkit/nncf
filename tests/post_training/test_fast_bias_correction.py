from abc import abstractmethod
from typing import List, TypeVar

import pytest

from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrectionParameters
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from tests.torch.ptq.helpers import get_min_max_and_fbc_algo_for_test

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestFBCAlgorithm:
    @staticmethod
    @abstractmethod
    def list_to_backend_type(data: List) -> TTensor:
        pass

    @staticmethod
    @abstractmethod
    def get_backend() -> FastBiasCorrectionAlgoBackend:
        pass

    @pytest.mark.parametrize(
        "bias_value, bias_shift, channel_axis, ref_shape",
        (
            ([1, 1], [0.1, 0.1], 1, [2]),
            ([[1, 1]], [0.1, 0.1], -1, [1, 2]),
            ([[1, 1]], [0.1, 0.1], 1, [1, 2]),
        ),
    )
    def test_reshape_bias_shift(self, bias_value, bias_shift, channel_axis, ref_shape):
        """
        Checks the result of the FastBiasCorrection.reshape_bias_shift method for backend specific datatype.
        """
        bias_value = self.list_to_backend_type(data=bias_value)
        bias_shift = self.list_to_backend_type(data=bias_shift)

        algo = FastBiasCorrection(FastBiasCorrectionParameters(number_samples=1,inplace_statistics=False))
        # pylint: disable=protected-access
        algo._backend_entity = self.get_backend()
        new_bias_shift = algo.reshape_bias_shift(bias_shift, bias_value, channel_axis)
        assert list(new_bias_shift.shape) == ref_shape

    @staticmethod
    @abstractmethod
    def get_model(with_bias, tmp_dir):
        pass

    @staticmethod
    @abstractmethod
    def get_dataset(model):
        pass

    @staticmethod
    @abstractmethod
    def check_bias(model, with_bias):
        pass

    @pytest.mark.parametrize("with_bias", (False, True))
    def test_fast_bias_correction_algo(self, with_bias, tmpdir):
        """
        Check working on fast bias correction algorithm and compare bias in quantized model with reference
        """
        model = self.get_model(with_bias, tmpdir)
        dataset = self.get_dataset(model)

        quantization_algorithm = get_min_max_and_fbc_algo_for_test()
        quantized_model = quantization_algorithm.apply(model, dataset=dataset)

        self.check_bias(quantized_model, with_bias)
