from abc import abstractmethod
from typing import Tuple

import pytest
from typing import List, TypeVar

from nncf.common.factory import NNCFGraphFactory
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from tests.torch.test_models.synthetic import ConcatSDPABlock
import torch

TModel = TypeVar("TModel")

class TemplateTestUnifiedScales:
    @property
    @abstractmethod
    def get_backend_specific_model(self, model: TModel) -> TModel:
        """
        Convert and return backend specific Model

        :param model: Model (for example in PT) to be converted to backend specific model
        :return: Backend specific Model
        """

    @pytest.mark.parametrize(
            "model_cls, unified_group, unified_group_nncf_network",
            (
                (ConcatSDPABlock, [['x', 'y']], [['/nncf_model_input_0', '/nncf_model_input_1']]),
            ),
        )
    def test_unified_groups(self, model_cls: TModel, unified_group: List[List[str]], unified_group_nncf_network: List[List[str]]):
        backend_model = self.get_backend_specific_model(model_cls())
        if isinstance(backend_model, torch.nn.Module) and not isinstance(backend_model, torch.fx.GraphModule):
            unified_group = unified_group_nncf_network

        nncf_graph = NNCFGraphFactory.create(backend_model)
        algo = MinMaxQuantization()
        algo._set_backend_entity(backend_model)
        algo._init_cache()
        _, groups = algo._find_quantization_target_points(backend_model, nncf_graph)
        assert [[target.target_node_name for target in groups] for groups in groups] == unified_group