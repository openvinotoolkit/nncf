# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import nncf.torch.graph.operator_metatypes as om


def test_all_torch_metatypes_are_registered() -> None:
    """
    Checks that every metatype with aliases defined in the torch operator_metatypes module
    is registered in PT_OPERATOR_METATYPES.
    """
    registered = set(om.PT_OPERATOR_METATYPES.registry_dict.values())
    defined = [
        cls
        for _, cls in inspect.getmembers(om, inspect.isclass)
        if issubclass(cls, om.PTOperatorMetatype)
        and cls not in (om.PTOperatorMetatype, om.PTOperatorSubtype, om.PTDepthwiseConvOperatorSubtype)
    ]

    assert defined, "No torch metatypes found in the module"
    not_registered = [cls.__name__ for cls in defined if cls not in registered]
    assert not not_registered, f"Metatypes are not registered in PT_OPERATOR_METATYPES: {not_registered}"
