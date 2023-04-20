import pytest
import numpy as np

from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from nncf.experimental.openvino_native.statistics.collectors import OVNoopReducer
from nncf.experimental.openvino_native.statistics.collectors import OVMinReducer
from nncf.experimental.openvino_native.statistics.collectors import OVMaxReducer
from nncf.experimental.openvino_native.statistics.collectors import OVAbsMaxReducer
from nncf.experimental.openvino_native.statistics.collectors import OVMeanReducer
from nncf.experimental.openvino_native.statistics.collectors import OVQuantileReducer
from nncf.experimental.openvino_native.statistics.collectors import OVAbsQuantileReducer
from nncf.experimental.openvino_native.statistics.collectors import OVBatchMeanReducer
from nncf.experimental.openvino_native.statistics.collectors import OVMeanPerChanelReducer
from nncf.experimental.openvino_native.statistics.collectors import OVNNCFCollectorTensorProcessor

from tests.experimental.common.test_reducers_and_aggregators import TemplateTestReducersAggreagtors


class TestReducersAggregators(TemplateTestReducersAggreagtors):
    @pytest.fixture
    def tensor_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_nncf_tensor(self, x: np.array):
        return OVNNCFTensor(x)

    @pytest.fixture(scope='module')
    def reducers(self):
        return {reducer.NAME: reducer for reducer in\
            [OVNoopReducer, OVMinReducer, OVMaxReducer, OVAbsMaxReducer, OVMeanReducer,
             OVQuantileReducer, OVAbsQuantileReducer, OVBatchMeanReducer, OVMeanPerChanelReducer]}

    def all_close(self, val, ref) -> bool:
        val_ = np.array(val)
        ref_ = np.array(ref)
        return np.allclose(val_, ref_) and val_.shape == ref_.shape
