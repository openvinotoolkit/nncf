from typing import Callable

from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType


class PTTargetPoint(TargetPoint):
    def __init__(self, target_type: TargetType, *,
                 target_node_name: NNCFNodeName,
                 input_port_id: int = None):
        super().__init__(target_type)
        self.target_type = target_type
        if self.target_type not in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION,
                                    TargetType.OPERATION_WITH_WEIGHTS,
                                    TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]:
            raise NotImplementedError("Unsupported target type: {}".format(target_type))

        self.target_node_name = target_node_name
        self.input_port_id = input_port_id

    def __eq__(self, other: 'PTTargetPoint'):
        return self.target_type == other.target_type and self.target_node_name == other.target_node_name

    def __str__(self):
        prefix = str(self.target_type)
        retval = prefix
        if self.target_type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION,
                                TargetType.OPERATION_WITH_WEIGHTS]:
            retval += " {}".format(self.target_node_name)
        elif self.target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.target_node_name)
        return retval

    def __hash__(self):
        return hash(str(self))


class PTInsertionCommand(TransformationCommand):
    def __init__(self, point: PTTargetPoint, fn: Callable,
                 priority: TransformationPriority = TransformationPriority.DEFAULT_PRIORITY):
        super().__init__(TransformationType.INSERT, point)
        self.fn = fn  # type: Callable
        self.priority = priority  # type: TransformationPriority

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # TODO: keep all TransformationCommands atomic, refactor TransformationLayout instead
        raise NotImplementedError()
