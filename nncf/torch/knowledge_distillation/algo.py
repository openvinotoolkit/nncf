from copy import deepcopy

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf import NNCFConfig
from nncf.torch.knowledge_distillation.knowledge_distillation_loss import KnowledgeDistillationLoss
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.api.compression import CompressionLoss, CompressionScheduler, CompressionStage
from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS


@COMPRESSION_ALGORITHMS.register('knowledge_distillation')
class KnowledgeDistillationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.kd_type = self._algo_config.get('type', None)
        if self.kd_type is None:
            raise ValueError('Type of KDLoss must be selected explicitly')

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        self.original_model = deepcopy(target_model.nncf_module)
        for param in self.original_model.parameters():
            param.requires_grad = False
        return PTTransformationLayout()

    def _build_controller(self, target_model):
        return KnowledgeDistillationController(target_model, self.original_model, self.kd_type)

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class KnowledgeDistillationController(PTCompressionAlgorithmController):
    def __init__(self, target_model, original_model, kd_type):
        super().__init__(target_model)
        original_model.train()
        self._scheduler = BaseCompressionScheduler()
        self._loss = KnowledgeDistillationLoss(target_model=target_model,
                                               original_model=original_model,
                                               kd_type=kd_type)

    def compression_stage(self) -> CompressionStage:
        """
        Returns level of compression. Should be used on saving best checkpoints to distinguish between
        uncompressed, partially compressed and fully compressed models.
        """
        return CompressionStage.FULLY_COMPRESSED

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
