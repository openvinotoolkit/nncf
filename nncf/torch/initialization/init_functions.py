from typing import Any
from typing import Callable
from typing import Union

import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nncf.config.structure import BNAdaptationInitArgs
from nncf.torch.initialization.data_loader import InitializingDataLoader
from nncf.torch.initialization.data_loader import wrap_dataloader_for_init
from nncf.torch.structures import AutoQPrecisionInitArgs
from nncf.torch.structures import QuantizationPrecisionInitArgs
from nncf.torch.structures import QuantizationRangeInitArgs


def default_criterion_fn(outputs: Any, target: Any, criterion: Any) -> torch.Tensor:
    return criterion(outputs, target)


def register_default_init_args(nncf_config: 'NNCFConfig',
                               train_loader: Union[torch.utils.data.DataLoader, InitializingDataLoader],
                               criterion: _Loss = None,
                               criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor] = None,
                               autoq_eval_fn: Callable[[torch.nn.Module, torch.utils.data.DataLoader], float] = None,
                               autoq_eval_loader: torch.utils.data.DataLoader = None,
                               device: str = None) -> 'NNCFConfig':
    nncf_config.register_extra_structs([QuantizationRangeInitArgs(data_loader=train_loader,
                                                                  device=device),
                                        BNAdaptationInitArgs(data_loader=wrap_dataloader_for_init(train_loader),
                                                             device=device)])

    if criterion:
        if not criterion_fn:
            criterion_fn = default_criterion_fn
        nncf_config.register_extra_structs([QuantizationPrecisionInitArgs(criterion_fn=criterion_fn,
                                                                          criterion=criterion,
                                                                          data_loader=train_loader,
                                                                          device=device)])

    if autoq_eval_fn:
        if not autoq_eval_loader:
            autoq_eval_loader = train_loader
        nncf_config.register_extra_structs([AutoQPrecisionInitArgs(data_loader=autoq_eval_loader,
                                                                   eval_fn=autoq_eval_fn,
                                                                   nncf_config=nncf_config)])

    return nncf_config
