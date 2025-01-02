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

from collections import namedtuple
from pathlib import Path
from typing import List

import jsonschema
import pytest

from nncf.config import NNCFConfig
from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.cross_fw.shared.paths import TEST_ROOT

GOOD_CONFIG_SOURCES = [
    PROJECT_ROOT / Path("examples/torch/classification/configs"),
    PROJECT_ROOT / Path("examples/torch/semantic_segmentation/configs"),
    PROJECT_ROOT / Path("examples/torch/object_detection/configs"),
    TEST_ROOT / Path("torch/data/configs"),
    TEST_ROOT / Path("torch/data/schema_validation_good_configs"),
    PROJECT_ROOT / Path("examples/tensorflow/classification/configs"),
    PROJECT_ROOT / Path("examples/tensorflow/segmentation/configs"),
    PROJECT_ROOT / Path("examples/tensorflow/object_detection/configs"),
    TEST_ROOT / Path("tensorflow/data/configs"),
    TEST_ROOT / Path("common/data/schema_validation_good_configs"),
]

BAD_CONFIG_SOURCES = [TEST_ROOT / Path("common/data/schema_validation_bad_configs")]

ConfigPathVsPassesSchemaVal = namedtuple("ConfigPathVsPassesSchemaVal", ("path", "should_pass"))
TEST_STRUCTS = []


def get_all_jsons_from_sources(source_directories_list: List[Path]) -> List[Path]:
    retval = []
    for source_dir in source_directories_list:
        files = source_dir.glob("**/*.json")
        retval += files
    return retval


good_config_files = get_all_jsons_from_sources(GOOD_CONFIG_SOURCES)
for file in good_config_files:
    TEST_STRUCTS.append(ConfigPathVsPassesSchemaVal(file, True))

bad_config_files = get_all_jsons_from_sources(BAD_CONFIG_SOURCES)
for file in bad_config_files:
    TEST_STRUCTS.append(ConfigPathVsPassesSchemaVal(file, False))


@pytest.fixture(
    name="config_test_struct",
    params=TEST_STRUCTS,
    ids=[str(struct.path.relative_to(PROJECT_ROOT)) for struct in TEST_STRUCTS],
)
def _config_test_struct(request):
    return request.param


def test_json_against_nncf_config_schema(config_test_struct):
    config_path, should_pass = config_test_struct
    if should_pass:
        _ = NNCFConfig.from_json(str(config_path))
    else:
        with pytest.raises(jsonschema.ValidationError):
            _ = NNCFConfig.from_json(str(config_path))
