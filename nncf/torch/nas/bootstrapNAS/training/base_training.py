from abc import ABC
from abc import abstractmethod

from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor


class BNASTrainingAlgorithm(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def set_stage(self, stage_desc: StageDescriptor):
        pass

    @abstractmethod
    def prepare_for_validation(self):
        pass

    @abstractmethod
    def get_total_num_epochs(self) -> int:
        pass


class BNASTrainingController(PTCompressionAlgorithmController, BNASTrainingAlgorithm, ABC):
    @property
    @abstractmethod
    def multi_elasticity_handler(self) -> MultiElasticityHandler:
        pass

    @property
    @abstractmethod
    def elasticity_controller(self):
        pass
