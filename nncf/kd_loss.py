from typing import List

from copy import deepcopy
from torch import nn
import torch
from functools import reduce
from nncf import NNCFConfig

from nncf.nncf_network import NNCFNetwork, InsertionCommand
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.compression_method_api import CompressionLevel
from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import PTCompressionLoss
from nncf.utils import objwalk


class KDLossCalculator(PTCompressionLoss):
    def __init__(self, original_model, scale, is_softmax):
        super().__init__()
        self.original_model = original_model
        self.original_model.train()
        self.scale = scale
        self.is_softmax = is_softmax
        if is_softmax:
            def kdloss_fn(ref_outputs, compressed_model_outputs):
                return -(nn.functional.log_softmax(compressed_model_outputs, dim=1) *
                         nn.functional.softmax(ref_outputs, dim=1)).mean() * (compressed_model_outputs.shape[1])
        else:
            def kdloss_fn(ref_outputs, compressed_model_outputs):
                mse = torch.nn.MSELoss()
                return mse(ref_outputs, compressed_model_outputs)
        self.kdloss_fn = kdloss_fn

    def forward(self, input_=None, target=None):
        # input_ is compressed model output
        # target is input
        if input_ is None or target is None:
            raise ValueError('KDLoss entries cannot be None. Check compression loss arguments.')

        def is_loss(obj):
            if not isinstance(obj, torch.Tensor):
                return False
            if obj.requires_grad:
                return True
            return False

        def tensors_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return [obj]
            else:
                return list(obj.values() if isinstance(obj, dict) else obj)
        with torch.no_grad():
            ref_outputs = self.original_model(target)
        ref_loss_outputs = tensors_to_list(objwalk(ref_outputs, is_loss, lambda x: x))
        print(f'kd loss outputs {ref_loss_outputs[0].size()}')
        print(f'KDModel parameters device {next(self.original_model.parameters()).device}')
        compressed_model_loss_outputs = tensors_to_list(objwalk(input_, is_loss, lambda x: x))
        # check for shapes zip is not reliable
        return self.scale * reduce(
            lambda kd_loss, loss_tensors: kd_loss + self.kdloss_fn(loss_tensors[0], loss_tensors[1]),
            zip(ref_loss_outputs, compressed_model_loss_outputs), 0)

    def statistics(self, quickly_collected_only=False):
        return {}


@COMPRESSION_ALGORITHMS.register('knowledge_distillation')
class KnowledgeDistillationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.scale = config.get('scale', 1)
        self.is_softmax = config.get('is_softmax', False)

    def _apply_to(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        self.original_model = deepcopy(target_model.nncf_module)
        #self.original_model = torch.nn.DataParallel(self.original_model)
        return []

    def build_controller(self, target_model):
        return KnowledgeDistillationController(target_model, self.original_model, self.scale, self.is_softmax)


class KnowledgeDistillationController(PTCompressionAlgorithmController):
    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL

    def __init__(self, target_model, original_model, scale, is_softmax):
        super().__init__(target_model)
        self._loss = KDLossCalculator(original_model, scale, is_softmax)

    def distributed(self):
        pass
