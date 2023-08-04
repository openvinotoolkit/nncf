from typing import Callable

import torch

from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.tensor import Tensor
from nncf.torch import no_nncf_trace


def create_register_input_hook(collector: TensorCollector) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Function to create regiter inputs hook function.

    :param collector: Collector to use in resulting hook.
    :return: Register inputs hook function.
    """


    def register_inputs_hook(x: torch.Tensor) -> torch.Tensor:
        """
        Register inputs hook function.

        :parameter x: tensor to register in hook.
        :return: tensor to register in hook.
        """
        with no_nncf_trace():
            collector.register_input_for_all_reducers(Tensor(x))
        return x


    return register_inputs_hook
