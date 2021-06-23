from copy import deepcopy
from functools import reduce, partial

from torch import nn
import torch

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf import NNCFConfig
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.dynamic_graph.op_input_processing import OperatorInput
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.api.compression import CompressionLevel, CompressionLoss, CompressionScheduler
from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionLoss

KD_MODULE_NAME = 'KD_FP32_MODULE'


class KnowledgeDistillationLoss(PTCompressionLoss):
    def __init__(self, target_model: NNCFNetwork, original_model, type_):
        super().__init__()
        original_model.train()
        if type_ == 'softmax':
            def kd_loss_fn(ref_outputs, compressed_model_outputs):
                return -(nn.functional.log_softmax(compressed_model_outputs, dim=1) *
                         nn.functional.softmax(ref_outputs, dim=1)).mean() * (compressed_model_outputs.shape[1])
        elif type_ == 'mse':
            def kd_loss_fn(ref_outputs, compressed_model_outputs):
                mse = torch.nn.MSELoss()
                return mse(ref_outputs, compressed_model_outputs)
        else:
            raise ValueError('Choose between mse/softmax options for Knowledge Distillation')
        self._kd_loss_handler = target_model.create_kd_loss_handler(original_model, partial(
            KnowledgeDistillationLoss.calculate_kd_loss,
            device=next(target_model.parameters()).device,
            kd_loss_fn=kd_loss_fn))

    @staticmethod
    def calculate_kd_loss(compressed_model_outputs, orig_model_outputs, device, kd_loss_fn):
        """
        Calculates KD Loss value from compressed_model_outputs and orig_model_outputs. First uses
        nested_object_paths_generator to unpack input containers and numerate contents inside them.
        Than checks compressed_model_outputs unpacked container for loss tensors (requires_grad=True)
        and maps extracted structure of loss tensors to orig_model_outputs.
        Finally computes KD loss with extracted loss tensors.

        :param compressed_model_outputs: Output tensors of compressed model can be any type of container with
            deterministic detour
        :param orig_model_outputs: Output tensors of original model (used for distillation) can be any type of 
            container with deterministic detour.
        :return: KDLoss value
        """

        compressed_model_outputs_struct = []
        orig_model_outputs_struct = []
        OperatorInput.nested_object_paths_generator([compressed_model_outputs], compressed_model_outputs_struct)
        OperatorInput.nested_object_paths_generator([orig_model_outputs], orig_model_outputs_struct)
        compressed_model_loss_nested_obj_paths = list(filter(lambda x: KnowledgeDistillationLoss._is_loss(x.getter()),
                                                             compressed_model_outputs_struct))
        compressed_model_loss_outputs = list(map(lambda x: x.getter(), compressed_model_loss_nested_obj_paths))
        match_fn = partial(KnowledgeDistillationLoss._match_func, to_match_with=compressed_model_loss_nested_obj_paths)
        orig_model_loss_outputs = list(map(lambda x: x.getter(), filter(match_fn, orig_model_outputs_struct)))
        if len(orig_model_loss_outputs) == 0 or len(compressed_model_loss_outputs) == 0:
            return torch.zeros([], device=device)
        return reduce(
            lambda kd_loss, loss_tensors: kd_loss + kd_loss_fn(loss_tensors[0], loss_tensors[1]),
            zip(orig_model_loss_outputs, compressed_model_loss_outputs), 0)

    @staticmethod
    def _is_loss(obj):
        if not isinstance(obj, torch.Tensor):
            return False
        if obj.requires_grad:
            return True
        return False

    @staticmethod
    def _match_func(obj, to_match_with):
        for x in to_match_with:
            if x.path == obj.path:
                return True
        return False

    def forward(self):
        loss = self._kd_loss_handler.get_kdloss()
        if isinstance(loss, int):
            return loss
        if len(loss) == 0:
            raise RuntimeError('Empty list of loss tensors for KDLoss. Most likely compression_ctrl.loss()'
                               ' was called while model was in eval mode')
        for i in range(len(loss)):
            loss[i] = loss[i].unsqueeze(0)
        output = torch.cat(loss).mean()
        self._kd_loss_handler.zero_kdloss()
        return output

    def statistics(self, quickly_collected_only=False):
        return {}


@COMPRESSION_ALGORITHMS.register('knowledge_distillation')
class KnowledgeDistillationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.type_ = config.get('type', None)
        if self.type_ is None:
            raise ValueError('You have to choose type of KDLoss explicitly')

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        self.original_model = deepcopy(target_model.nncf_module)
        for param in self.original_model.parameters():
            param.requires_grad = False
        return PTTransformationLayout()

    def build_controller(self, target_model):
        return KnowledgeDistillationController(target_model, self.original_model, self.type_)

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class KnowledgeDistillationController(PTCompressionAlgorithmController):
    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL

    def __init__(self, target_model, original_model, type_):
        super().__init__(target_model)
        original_model.train()
        self._scheduler = BaseCompressionScheduler()
        self._loss = KnowledgeDistillationLoss(target_model=target_model, original_model=original_model, type_=type_)

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def distributed(self):
        pass
