import pytest
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from abc import abstractmethod

from nncf.experimental.common.tensor_statistics.collectors import TensorType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.common.tensor_statistics.collectors import TensorAggregatorBase
from nncf.experimental.common.tensor_statistics.collectors import MergedTensorCollector
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MedianAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanNoOutliersAggregator
from nncf.experimental.common.tensor_statistics.collectors import MedianNoOutliersAggregator


class TemplateTestReducersAggreagtors:
    @abstractmethod
    def get_nncf_tensor(self, x: np.array):
        pass

    @pytest.fixture
    @abstractmethod
    def tensor_processor(self):
        pass

    @pytest.fixture
    @abstractmethod
    def test_params(self):
        pass

    NO_OUTLIERS_TEST_PARAMS = [
        (MeanAggregator, True, 1, [-49992.0]),
        (MedianAggregator, True, 1, [8.5]),
        (MeanAggregator, False, 1,
         [4.5,  9.0, -9e5,  9.0,  9.0,  9.0,
          13.5,  18.0,  4.5e5]),
        (MedianAggregator, False, 1,
         [4.5,  9.0, -9e5,  9.0,  9.0,  9.0,
          13.5,  18.0,  4.5e5]),

        (MeanAggregator, True, 2,
         [ 9.0,  12.0, -1.49997e+05]),
        (MedianAggregator, True, 2,
         [ 7.5, 11. ,  9. ]),
        (MeanAggregator, False, 2,
         [[ 4.5,  9.0, -9.e5],
          [ 9.0,  9.0,  9.0],
          [ 13.5,  18.0,  4.5e5]]),
        (MedianAggregator, False, 2,
         [[ 4.5,  9.0, -9.e5],
          [ 9.0,  9.0,  9.0],
          [ 13.5,  18.0,  4.5e5]]),

        (MeanAggregator, True, 3,
         [[ 4.5,  9.0, -9.e5],
          [ 9.0,  9.0,  9.0],
          [ 13.5,  18.0,  4.5e5]]),
        (MedianAggregator, True, 3,
         [[ 4.5,  9.0, -9.e5],
          [ 9.0,  9.0,  9.0],
          [ 13.5,  18.0,  4.5e5]]),
        (MeanAggregator, False, 3,
         [[ 4.5,  9.0, -9.e5],
          [ 9.0,  9.0,  9.0],
          [ 13.5,  18.0,  4.5e5]]),
        (MedianAggregator, False, 3,
         [[ 4.5,  9.0, -9.e5],
          [ 9.0,  9.0,  9.0],
          [ 13.5,  18.0,  4.5e5]]),


        (MeanNoOutliersAggregator, True, 1, [-39991.77142857]),
        (MedianNoOutliersAggregator, True, 1, [8.5]),
        (MeanNoOutliersAggregator, False, 1,
         [4.5,  9.0, -9e5,  9.0,  9.0,  9.0,
          13.5,  18.0,  4.5e5]),
        (MedianNoOutliersAggregator, False, 1,
         [4.5,  9.0, -9e5,  9.0,  9.0,  9.0,
          13.5,  18.0,  4.5e5]),

        (MeanNoOutliersAggregator, True, 2,
         [ 8.68181818, 11.13043478,  -1.27269455e+05]),
        (MedianNoOutliersAggregator, True, 2,
         [ 7.5, 10. ,  9. ]),
        (MeanNoOutliersAggregator, False, 2,
         [2.2857142857142856, 4.571428571428571, 6.857142857142857,
          9.142857142857142, 11.428571428571429, 13.714285714285714,
          16.0, 18.285714285714285]),
        (MedianNoOutliersAggregator, False, 2, [2.0, 4.0, 6.0, 8.0, 10.0,
                                      12.0, 14.0, 16.0]),

        (MeanNoOutliersAggregator, True, 3,
         [2.2857142857142856, 4.571428571428571, 6.857142857142857,
          9.142857142857142, 11.428571428571429, 13.714285714285714,
          16.0, 18.285714285714285]),
        (MedianNoOutliersAggregator, True, 3, [2.0, 4.0, 6.0, 8.0, 10.0,
                                      12.0, 14.0, 16.0]),
        (MeanNoOutliersAggregator, False, 3,
         [2.2857142857142856, 4.571428571428571, 6.857142857142857,
          9.142857142857142, 11.428571428571429, 13.714285714285714,
          16.0, 18.285714285714285]),
        (MedianNoOutliersAggregator, False, 3, [2.0, 4.0, 6.0, 8.0, 10.0,
                                      12.0, 14.0, 16.0]),
    ][12:]

    @pytest.mark.parametrize('aggregator_cls,use_per_sample_stats,dims,refs', NO_OUTLIERS_TEST_PARAMS)
    def test_no_outliers_agggregators(self, aggregator_cls, refs, tensor_processor,
                                      dims, use_per_sample_stats):
        input_ = np.array([1, 2, -200000, 2, 2, 2, 3, 4, 100000])
        if dims == 2:
            input_ = input_.reshape((3, 3))
        if dims == 3:
            input_ = input_.reshape((1, 3, 3))
        aggregator = aggregator_cls(tensor_processor, use_per_sample_stats)
        for i in range(1, 9):
            aggregator.register_reduced_input(self.get_nncf_tensor(input_ * i))
        ret_val = aggregator.aggregate()
        assert np.allclose(ret_val, refs)
