from collections import UserDict

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor import TensorType

from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint


class StatisticPoint:
    def __init__(self, target_point: ONNXTargetPoint, tensor_collector: TensorStatisticCollectorBase, algorithm):
        """
        StatisticPoint is attached to the output of the node
        """
        self.target_point = target_point
        self.algorithm_to_tensor_collector = {algorithm: tensor_collector}

    def __eq__(self, other):
        if self.target_point == other.target_point and \
                self.algorithm_to_tensor_collector == other.self.algorithm_to_tensor_collector:
            return True
        return False

    def register_tensor(self, x: TensorType):
        for tensor_collector in self.algorithm_to_tensor_collector.values():
            tensor_collector.register_input(x)


class StatisticPointsContainer(UserDict):
    def add_statistic_point(self, statistic_point: StatisticPoint):
        if statistic_point.target_point.target_node_name not in self.data:
            self.data[statistic_point.target_point.target_node_name] = statistic_point
