import pytest
import numpy as np

from nncf.experimental.openvino_native.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from tests.experimental.common.test_reducers_and_aggregators import TemplateTestReducersAggreagtors


class TestReducersAggregators(TemplateTestReducersAggreagtors):
    @pytest.fixture
    def tensor_processor(self):
        return OVNNCFCollectorTensorProcessor

    def test_params(self):
        pass

    def get_nncf_tensor(self, x: np.array):
        return OVNNCFTensor(x)