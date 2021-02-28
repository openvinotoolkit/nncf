import functools
from enum import Enum
from typing import Callable

from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext


@functools.total_ordering
class OperationPriority(Enum):
    DEFAULT_PRIORITY = 0
    FP32_TENSOR_STATISTICS_OBSERVATION = 1
    SPARSIFICATION_PRIORITY = 2
    QUANTIZATION_PRIORITY = 11
    PRUNING_PRIORITY = 1

    def __lt__(self, other):
        # pylint: disable=comparison-with-callable
        return self.value < other.value


class InsertionType(Enum):
    OPERATOR_PRE_HOOK = 0
    OPERATOR_POST_HOOK = 1
    NNCF_MODULE_PRE_OP = 2
    NNCF_MODULE_POST_OP = 3

    def __eq__(self, other):
        # pylint: disable=comparison-with-callable
        if isinstance(other, InsertionType):
            return self.value == other.value
        return self.value == other


class InsertionPoint:
    def __init__(self, insertion_type: InsertionType, *,
                 ia_op_exec_context: InputAgnosticOperationExecutionContext = None,
                 module_scope: 'Scope' = None,
                 input_port_id: int = None):
        self.insertion_type = insertion_type
        if self.insertion_type in [InsertionType.NNCF_MODULE_PRE_OP, InsertionType.NNCF_MODULE_POST_OP]:
            if module_scope is None:
                raise ValueError("Should specify module scope for module pre- and post-op insertion points!")

        if self.insertion_type in [InsertionType.OPERATOR_PRE_HOOK, InsertionType.OPERATOR_POST_HOOK]:
            if ia_op_exec_context is None:
                raise ValueError("Should specify an operator's InputAgnosticOperationExecutionContext "
                                 "for operator pre- and post-hook insertion points!")
        self.module_scope = module_scope
        self.ia_op_exec_context = ia_op_exec_context
        self.input_port_id = input_port_id

    def __eq__(self, other: 'InsertionPoint'):
        return self.insertion_type == other.insertion_type and self.ia_op_exec_context == other.ia_op_exec_context \
               and self.input_port_id == other.input_port_id and self.module_scope == other.module_scope

    def __str__(self):
        prefix = str(self.insertion_type)
        retval = prefix
        if self.insertion_type in [InsertionType.NNCF_MODULE_PRE_OP, InsertionType.NNCF_MODULE_POST_OP]:
            retval += " {}".format(self.module_scope)
        elif self.insertion_type in [InsertionType.OPERATOR_PRE_HOOK, InsertionType.OPERATOR_POST_HOOK]:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.ia_op_exec_context)
        return retval

    def __hash__(self):
        return hash(str(self))


class PTInsertionCommand:
    def __init__(self, point: InsertionPoint, fn: Callable,
                 priority: OperationPriority = OperationPriority.DEFAULT_PRIORITY):
        self.insertion_point = point  # type: InsertionPoint
        self.fn = fn  # type: Callable
        self.priority = priority  # type: OperationPriority


