from copy import deepcopy

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.graph.transformations.layout import PTTransformationLayout
from torch import nn
import torch
from functools import reduce, partial
from nncf import NNCFConfig

from nncf.nncf_network import NNCFNetwork
from nncf.dynamic_graph.utils import nested_object_paths_generator
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.api.compression import CompressionLevel, CompressionLoss, CompressionScheduler
from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import PTCompressionLoss

KD_MODULE_NAME = 'KD_FP32_MODULE'


class KDLossCalculator(PTCompressionLoss):
    def __init__(self, target_model: NNCFNetwork, original_model, scale, type):
        super().__init__()
        self._target_model = target_model
        self.original_model = original_model
        self.original_model.train()
        self.scale = scale
        if type == 'softmax':
            def kdloss_fn(ref_outputs, compressed_model_outputs):
                return -(nn.functional.log_softmax(compressed_model_outputs, dim=1) *
                         nn.functional.softmax(ref_outputs, dim=1)).mean() * (compressed_model_outputs.shape[1])
        elif type == 'mse':
            def kdloss_fn(ref_outputs, compressed_model_outputs):
                mse = torch.nn.MSELoss()
                return mse(ref_outputs, compressed_model_outputs)
        else:
            raise ValueError('Choose between mse/softmax options for Knowledge Distillation')
        self.kdloss_fn = kdloss_fn

    def calculate_kd_loss(self, compressed_model_outputs, orig_model_outputs):
        compressed_model_outputs_struct = []
        orig_model_outputs_struct = []
        nested_object_paths_generator([compressed_model_outputs], compressed_model_outputs_struct)
        nested_object_paths_generator([orig_model_outputs], orig_model_outputs_struct)
        compressed_model_loss_nested_obj_paths = list(filter(lambda x: self._is_loss(x.getter()), compressed_model_outputs_struct))
        compressed_model_loss_outputs = list(map(lambda x: x.getter(), compressed_model_loss_nested_obj_paths))
        orig_model_loss_outputs = list(map(lambda x: x.getter(), filter(
            partial(self.match_func, to_match_with=compressed_model_loss_nested_obj_paths), orig_model_outputs_struct)))
        if len(orig_model_outputs) == 0 or len(compressed_model_loss_outputs) == 0:
            return torch.zeros([], device=next(self._target_model.parameters()).device)
        return self.scale * reduce(
            lambda kd_loss, loss_tensors: kd_loss + self.kdloss_fn(loss_tensors[0], loss_tensors[1]),
            zip(orig_model_loss_outputs, compressed_model_loss_outputs), 0)

    @staticmethod
    def _is_loss(obj):
        if not isinstance(obj, torch.Tensor):
            return False
        if obj.requires_grad:
            return True
        return False

    @staticmethod
    def match_func(obj, to_match_with):
        for x in to_match_with:
            if x.path == obj.path:
                return True
        return False

    def forward(self, input_=None, target=None):
        loss = self._target_model.get_kdloss()
        if isinstance(loss, int):
            return loss
        for i in range(len(loss)):
            loss[i] = loss[i].unsqueeze(0)
        output = torch.cat(loss).mean()
        self._target_model.zero_kdloss()
        return output

    def statistics(self, quickly_collected_only=False):
        return {}


@COMPRESSION_ALGORITHMS.register('knowledge_distillation')
class KnowledgeDistillationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.scale = config.get('scale', 1)
        self.type = config.get('type', None)
        if self.type is None:
            raise ValueError('You have to choose type of KDLoss explicitly')

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        self.original_model = deepcopy(target_model.nncf_module)
        return PTTransformationLayout()

    def build_controller(self, target_model):
        return KnowledgeDistillationController(target_model, self.original_model, self.scale, self.type)


class KnowledgeDistillationController(PTCompressionAlgorithmController):
    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL

    def __init__(self, target_model, original_model, scale, type):
        super().__init__(target_model)
        original_model.train()
        self._scheduler = BaseCompressionScheduler()
        self._loss = KDLossCalculator(target_model=target_model, original_model=original_model, scale=scale, type=type)
        target_model.enable_knowledge_distillation(original_model, self._loss.calculate_kd_loss)

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def distributed(self):
        pass
