from abc import ABC
from abc import abstractmethod

from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor


class BNASTrainingAlgorithm(ABC):
    """
    Base training algorithm for supernet-based NAS methods.
    """

    @abstractmethod
    def step(self) -> None:
        """
        Should be called at the beginning of each training step for activation some Subnet(s).
        """

    @abstractmethod
    def set_stage(self, stage_desc: StageDescriptor) -> None:
        """
        Set a new training stage with parameters from a given stage descriptor

        :param stage_desc: describes parameters of the training stage that should be enabled
        """

    @abstractmethod
    def prepare_for_validation(self) -> None:
        """
        Performs some action on active subnet or supernet before validation. For instance, it can be the batchnorm
        adaptation to achieve the best accuracy on validation.
        """

    @abstractmethod
    def get_total_num_epochs(self) -> int:
        """
        Returns total number of epochs required for the supernet training.

        :return: number of epochs
        """


class BNASTrainingController(PTCompressionAlgorithmController, BNASTrainingAlgorithm, ABC):
    @property
    @abstractmethod
    def multi_elasticity_handler(self) -> MultiElasticityHandler:
        """
        Gets access to multi elasticity handler to perform some actions with supernet or subnets.

        :return: multi elasticity handler
        """

    @property
    @abstractmethod
    def elasticity_controller(self) -> ElasticityController:
        """
        Gets access to elasticity controller. Usually it's needed for saving its state for further resuming in the
        search part.

        :return: elasticity controller
        """
