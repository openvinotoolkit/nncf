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
        self.algorithm_to_tensor_collectors = {algorithm: [tensor_collector]}

    def __eq__(self, other):
        if self.target_point == other.target_point and \
                self.algorithm_to_tensor_collectors == other.self.algorithm_to_tensor_collectors:
            return True
        return False

    def register_tensor(self, x: TensorType):
        for tensor_collectors in self.algorithm_to_tensor_collectors.values():
            for tensor_collector in tensor_collectors:
                tensor_collector.register_input(x)


class StatisticPointsContainer(UserDict):
    def add_statistic_point(self, statistic_point: StatisticPoint):
        target_node_name = statistic_point.target_point.target_node_name
        if target_node_name not in self.data:
            self.data[target_node_name] = [statistic_point]
        else:
            for _statistic_point in self.data[target_node_name]:
                if _statistic_point.target_point == statistic_point.target_point:
                    for algorithm in statistic_point.algorithm_to_tensor_collectors.keys():
                        if algorithm in _statistic_point.algorithm_to_tensor_collectors:
                            _statistic_point.algorithm_to_tensor_collectors[algorithm].extend(
                                statistic_point.algorithm_to_tensor_collectors[algorithm])
                            return
                        _statistic_point.algorithm_to_tensor_collectors[
                            algorithm] = statistic_point.algorithm_to_tensor_collectors[algorithm]
                        return
            self.data[target_node_name].append(statistic_point)
