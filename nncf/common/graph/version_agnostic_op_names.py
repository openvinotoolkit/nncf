"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from nncf.common.utils.backend import __nncf_backend__

def get_version_agnostic_name(version_specific_name: str):
    """
    Returns the operator name that does not depend on the framework version.

    For example, different framework versions have different names for
    the same operator:
        'operator_name_v1' : framework version v1,
        'operator_name_v2' : framework version v2

    get_version_agnostic_name('operator_name_v1') == get_version_agnostic_name('operator_name_v2')

    :param version_specific_name: The operator name
    :return: The framework version agnostic operator name
    """
    def get_func_impl():
        if __nncf_backend__ == 'Torch':
            from nncf.graph.version_agnostic_op_names \
                import get_version_agnostic_name as torch_fn_impl
            return torch_fn_impl

        def default_func_impl(version_specific_name: str):
            return version_specific_name

        return default_func_impl

    func = get_func_impl()
    return func(version_specific_name)
