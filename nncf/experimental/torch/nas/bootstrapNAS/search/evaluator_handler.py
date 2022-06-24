from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar

from nncf.common.utils.logger import logger as nncf_logger

BaseEvaluatorType = TypeVar('BaseEvaluatorType')
ElasticControllerType = TypeVar('ElasticControllerType')
SearchParametersType = TypeVar('SearchParametersType')

class BaseEvaluatorHandler:
    """
    An interface for handling an evaluator. Evaluator handlers initialize the underlying
    evaluator and can be used to update the evaluator's properties.

    """
    def __init__(self, evaluator: BaseEvaluatorType, elasticity_ctr: ElasticControllerType):
        """
        Initializes the evaluator handler

        :param evaluator: An interface for collecting measurements at a target device.
        :param elasticity_ctr: interface to manage the elasticity of the super-network.
        """
        self.evaluator = evaluator
        self.elasticity_ctrl = elasticity_ctr
        self.elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        self.input_model_value = evaluator.evaluate_subnet()

    @property
    def name(self):
        return self.evaluator.name

    @property
    def current_value(self):
        return self.evaluator.current_value

    def retrieve_from_cache(self, subnet_config_repr: Tuple[float, ...]) -> Tuple[bool, float]:
        return self.evaluator.retrieve_from_cache(subnet_config_repr)

    def evaluate_and_add_to_cache_from_pymoo(self, pymoo_repr: Tuple[float, ...]) -> float:
        return self.evaluator.evaluate_and_add_to_cache_from_pymoo(pymoo_repr)

    def export_cache_to_csv(self, cache_file_path: str) -> NoReturn:
        self.evaluator.export_cache_to_csv(cache_file_path)


class EfficiencyEvaluatorHandler(BaseEvaluatorHandler):
    """
    An interface for handling efficiency evaluators
    """


class AccuracyEvaluatorHandler(BaseEvaluatorHandler):
    """
    An interface for handling accuracy evaluators

    """
    def __init__(self, accuracy_evaluator, elasticity_ctrl, ref_acc: Optional[float] = 100):
        super().__init__(accuracy_evaluator, elasticity_ctrl)
        self._ref_acc = ref_acc

    @property
    def ref_acc(self) -> float:
        """
        :return: reference accuracy
        """
        return self._ref_acc

    @ref_acc.setter
    def ref_acc(self, val: float) -> NoReturn:
        """
        :param val: value to update the reference accuracy value.
        :return:
        """
        self._ref_acc = val

    def update_reference_accuracy(self, search_params: SearchParametersType) -> NoReturn:
        """
        Update reference accuracy of the search algorithm

        :param search_params: parameters of the search algorithm
        :return:
        """
        self.ref_acc = search_params.ref_acc
        if self.input_model_value > self.ref_acc - 0.01 or self.input_model_value < self.ref_acc + 0.01:
            nncf_logger.warning("Accuracy obtained from evaluation {value} differs from "
                                        "reference accuracy {ref_acc}".format(value=self.input_model_value,
                                                                              ref_acc=self.ref_acc))
            if self.ref_acc == -1:
                nncf_logger.info("Adjusting reference accuracy to accuracy obtained from evaluation")
                self.ref_acc = self.input_model_value
            else:
                if self.ref_acc >= 100:
                    nncf_logger.error("Reference accuracy value is invalid: {val}".format(
                        val=self.ref_acc))
                nncf_logger.info("Using reference accuracy.")
                self.input_model_value = self.ref_acc
        search_params.ref_acc = self.ref_acc
