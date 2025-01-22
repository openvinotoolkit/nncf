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
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from nncf.torch.algo_selector import NoCompressionAlgorithmController
from tests.cross_fw.sdl.symlinks import FileOpeningEntrypointProvider
from tests.cross_fw.sdl.symlinks import check_symlinks_are_not_followed


class PTCompressionControllerExportEntrypoint(FileOpeningEntrypointProvider):
    def get_entrypoint(self) -> Callable[[str], Any]:
        mock_pt_model = torch.nn.Module()
        ctrl = NoCompressionAlgorithmController(mock_pt_model)
        return ctrl.export_model


@pytest.mark.parametrize("file_opening_entrypoint", [PTCompressionControllerExportEntrypoint()])
def test_symlinks_are_not_followed(tmp_path: Path, file_opening_entrypoint: FileOpeningEntrypointProvider):
    entrypoint = file_opening_entrypoint.get_entrypoint()
    check_symlinks_are_not_followed(tmp_path, entrypoint)
