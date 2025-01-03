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

import re

import jsonschema
import pytest

from nncf.config.schema import validate_accuracy_aware_training_schema
from nncf.config.schema import validate_single_compression_algo_schema
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA

MOCK_ALGO_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {"const": "mock_algo_name"},
        "mock_property_1": with_attributes(BOOLEAN, description="first mock property"),
        "mock_property_2": with_attributes(BOOLEAN, description="second mock property"),
        "mode": with_attributes(STRING, description="mock mode"),
    },
    "required": ["mock_property_1"],
    "additionalProperties": False,
}


@pytest.fixture
def mock_algo_dict():
    return {
        "algorithm": "mock_algo_name",
        "mock_property_1": True,
        "mock_property_2": True,
        "mode": "mock_mode",
    }


MOCK_REF_VS_ALGO_SCHEMA = {"mock_algo_name": MOCK_ALGO_SCHEMA}
MOCK_ALGO_NAME_VS_README_URL = {"mock_algo_name": "docs/mock_algo_name.md"}

MOCK_ACCURACY_AWARE_TRAINING_SCHEMA = {
    "properties": {
        "algorithm": {"const": "mock_algo_name"},
        "mock_property_1": with_attributes(BOOLEAN, description="first mock property"),
        "mock_property_2": with_attributes(BOOLEAN, description="second mock property"),
    },
    "required": ["mock_property_2"],
}
MOCK_ACCURACY_AWARE_MODES_VS_SCHEMA = {"mock_mode": MOCK_ACCURACY_AWARE_TRAINING_SCHEMA}


def test_validate_single_compression_algo_schema_valid(mock_algo_dict):
    validate_single_compression_algo_schema(mock_algo_dict, MOCK_REF_VS_ALGO_SCHEMA)


def test_validate_single_compression_algo_schema_invalid_name(mock_algo_dict):
    with pytest.raises(jsonschema.ValidationError):
        mock_algo_dict["algorithm"] = "invalid_algo_name"
        validate_single_compression_algo_schema(mock_algo_dict, MOCK_REF_VS_ALGO_SCHEMA)


def test_validate_single_compression_algo_schema_missing_property(mock_algo_dict):
    with pytest.raises(jsonschema.ValidationError):
        del mock_algo_dict["mock_property_1"]
        validate_single_compression_algo_schema(mock_algo_dict, MOCK_REF_VS_ALGO_SCHEMA)


def test_validate_single_compression_algo_schema_additional_property(mock_algo_dict):
    with pytest.raises(jsonschema.ValidationError):
        mock_algo_dict["invalid_property"] = True
        validate_single_compression_algo_schema(mock_algo_dict, MOCK_REF_VS_ALGO_SCHEMA)


def test_validate_single_compression_with_readme_url(mock_algo_dict, monkeypatch):
    with pytest.raises(
        jsonschema.ValidationError,
        match=re.compile("or to the algorithm documentation for examples of the configs:"),
    ):
        monkeypatch.setattr("nncf.config.schema.ALGO_NAME_VS_README_URL", MOCK_ALGO_NAME_VS_README_URL)
        mock_algo_dict["invalid_property"] = True
        validate_single_compression_algo_schema(mock_algo_dict, MOCK_REF_VS_ALGO_SCHEMA)


def test_validate_accuracy_aware_training_schema_valid(mock_algo_dict, monkeypatch):
    monkeypatch.setattr(
        "nncf.config.schema.ACCURACY_AWARE_TRAINING_SCHEMA",
        MOCK_ACCURACY_AWARE_TRAINING_SCHEMA,
    )
    monkeypatch.setattr(
        "nncf.config.schema.ACCURACY_AWARE_MODES_VS_SCHEMA",
        MOCK_ACCURACY_AWARE_MODES_VS_SCHEMA,
    )
    validate_accuracy_aware_training_schema(mock_algo_dict)


def test_validate_accuracy_aware_training_schema_invalid_mode(mock_algo_dict, monkeypatch):
    with pytest.raises(jsonschema.ValidationError, match=re.compile("Incorrect Accuracy Aware mode")):
        monkeypatch.setattr(
            "nncf.config.schema.ACCURACY_AWARE_TRAINING_SCHEMA",
            MOCK_ACCURACY_AWARE_TRAINING_SCHEMA,
        )
        monkeypatch.setattr(
            "nncf.config.schema.ACCURACY_AWARE_MODES_VS_SCHEMA",
            MOCK_ACCURACY_AWARE_MODES_VS_SCHEMA,
        )
        mock_algo_dict["mode"] = "invalid_mode"
        validate_accuracy_aware_training_schema(mock_algo_dict)


def test_validate_accuracy_aware_training_schema_missing_property(mock_algo_dict, monkeypatch):
    with pytest.raises(jsonschema.ValidationError):
        monkeypatch.setattr(
            "nncf.config.schema.ACCURACY_AWARE_TRAINING_SCHEMA",
            MOCK_ACCURACY_AWARE_TRAINING_SCHEMA,
        )
        monkeypatch.setattr(
            "nncf.config.schema.ACCURACY_AWARE_MODES_VS_SCHEMA",
            MOCK_ACCURACY_AWARE_MODES_VS_SCHEMA,
        )
        del mock_algo_dict["mock_property_2"]
        validate_accuracy_aware_training_schema(mock_algo_dict)
