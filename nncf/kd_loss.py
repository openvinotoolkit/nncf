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
    def __init__(self, target_model, original_model, scale, is_softmax):
        super().__init__()
        self._target_model = target_model
        self.original_model = original_model
        self.original_model.train()
        self.scale = scale
        self.is_softmax = is_softmax
        # ToDo: Make user to choose type of loss explicitely
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

        def match_func(obj, to_match_with):
            for x in to_match_with:
                if x.path == obj.path:
                    return True
            return False

        # get outputs from nncf_model
        # nncf_model.get_register_modules_outputs()
        #with torch.no_grad():
        #    ref_outputs_0 = self.original_model(target)

        ref_outputs = self._target_model.get_registered_modules_for_parallel_exec_outputs(KD_MODULE_NAME)

        compressed_model_outputs = []
        orig_model_outputs = []
        nested_object_paths_generator([input_], compressed_model_outputs)
        nested_object_paths_generator([ref_outputs], orig_model_outputs)
        compressed_model_loss_nested_obj_paths = list(filter(lambda x: is_loss(x.getter()), compressed_model_outputs))
        compressed_model_loss_outputs = list(map(lambda x: x.getter(), compressed_model_loss_nested_obj_paths))
        orig_model_loss_outputs = list(map(lambda x: x.getter(), filter(partial(match_func, to_match_with=compressed_model_loss_nested_obj_paths), orig_model_outputs)))
        # check for shapes. zip is not reliable
        return self.scale * reduce(
            lambda kd_loss, loss_tensors: kd_loss + self.kdloss_fn(loss_tensors[0], loss_tensors[1]),
            zip(orig_model_loss_outputs, compressed_model_loss_outputs), 0)

    def statistics(self, quickly_collected_only=False):
        return {}


@COMPRESSION_ALGORITHMS.register('knowledge_distillation')
class KnowledgeDistillationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.scale = config.get('scale', 1)
        self.is_softmax = config.get('is_softmax', False)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        self.original_model = deepcopy(target_model.nncf_module)
        #target_model.register_module_for_parallel_execution(self.original_model, KD_MODULE_NAME, is_traced=False)
        #self.original_model = torch.nn.DataParallel(self.original_model)
        return PTTransformationLayout()

    def build_controller(self, target_model):
        return KnowledgeDistillationController(target_model, self.original_model, self.scale, self.is_softmax)


class KnowledgeDistillationController(PTCompressionAlgorithmController):
    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL

    def __init__(self, target_model, original_model, scale, is_softmax):
        super().__init__(target_model)
        original_model.train()
        target_model.register_module_for_parallel_exec(original_model, KD_MODULE_NAME, is_traced=False, no_grad=True)
        self._scheduler = BaseCompressionScheduler()
        self._loss = KDLossCalculator(target_model=target_model, original_model=original_model, scale=scale, is_softmax=is_softmax)

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def distributed(self):
        pass
