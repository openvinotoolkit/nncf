from typing import Callable

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.graph.graph import InputAgnosticOperationExecutionContext


class PTTargetPoint(TargetPoint):
    def __init__(self, target_type: TargetType, *,
                 ia_op_exec_context: InputAgnosticOperationExecutionContext = None,
                 module_scope: 'Scope' = None,
                 input_port_id: int = None):
        super().__init__(target_type)
        self.target_type = target_type
        if self.target_type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION,
                                TargetType.OPERATION_WITH_WEIGHTS]:
            if module_scope is None:
                raise ValueError("Should specify module scope for module pre- and post-op insertion points!")

        elif self.target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]:
            if ia_op_exec_context is None:
                raise ValueError("Should specify an operator's InputAgnosticOperationExecutionContext "
                                 "for operator pre- and post-hook insertion points!")
        else:
            raise NotImplementedError("Unsupported target type: {}".format(target_type))

        self.module_scope = module_scope
        self.ia_op_exec_context = ia_op_exec_context
        self.input_port_id = input_port_id

    def __eq__(self, other: 'PTTargetPoint'):
        return self.target_type == other.target_type and self.ia_op_exec_context == other.ia_op_exec_context \
               and self.input_port_id == other.input_port_id and self.module_scope == other.module_scope

    def __str__(self):
        prefix = str(self.target_type)
        retval = prefix
        if self.target_type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION,
                                TargetType.OPERATION_WITH_WEIGHTS]:
            retval += " {}".format(self.module_scope)
        elif self.target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.ia_op_exec_context)
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
