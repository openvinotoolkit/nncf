from typing import Optional

import torch.nn

from nncf.torch.dynamic_graph.scope import Scope


def get_module_by_scope(model: torch.nn.Module, scope: Scope) -> Optional[torch.nn.Module]:
    curr_module = model
    for scope_element in scope[1:]:  # omit first scope element which corresponds to base module
        if scope_element.calling_field_name is None:
            # The module used is being created in-place every time and never stored in the model,
            # happens for nn.Softmax in BERT implementations.
            return None
        # pylint: disable=protected-access
        next_module = curr_module._modules.get(scope_element.calling_field_name)
        if next_module is None:
            raise RuntimeError(
                "Could not find a {} module member in {} module of scope {} during node search".format(
                    scope_element.calling_field_name, scope_element.calling_module_class_name, str(scope)
                )
            )
        curr_module = next_module
    return curr_module
