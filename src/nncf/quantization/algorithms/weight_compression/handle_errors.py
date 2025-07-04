# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import nncf


def handle_invalid_group_size_error(first_caught_exception: nncf.InvalidGroupSizeError, node_names: list[str]) -> None:
    """
    Handles the InvalidGroupSizeError by generating a detailed error message and re-raising the exception.

    :param first_caught_exception: The first InvalidGroupSizeError instance that was caught.
    :param node_names: The list of node names where the error occurred.
        Used to suggest adding them to the ignored scope.
    :raises nncf.InvalidGroupSizeError: Re-raises the exception with an enhanced message suggesting corrective actions.
    """
    names = ",".join(f'"{name}"' for name in node_names)
    msg = (
        f"Compression of the '{node_names[0]}' layer failed with the following error:\n{first_caught_exception}\n"
        "Ensure that the group size is divisible by the channel size, "
        "or include this node and others with similar issues in the ignored scope:\n"
        f"nncf.compress_weight(\n\t..., \n\tignored_scope=IgnoredScope(names=[{names}]\n\t)\n)"
    )
    raise nncf.InvalidGroupSizeError(msg) from first_caught_exception
