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
from typing import Any, Dict, List, Optional

NUMBER = {"type": "number"}
STRING = {"type": "string"}
BOOLEAN = {"type": "boolean"}
ARRAY_OF_NUMBERS = {"type": "array", "items": NUMBER}
ARRAY_OF_STRINGS = {"type": "array", "items": STRING}


def annotated_enum(names_vs_description: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    retval_list = []
    for name, descr in names_vs_description.items():
        retval_list.append({"const": name, "title": name, "description": descr})
    return {"oneOf": retval_list}


def make_string_or_array_of_strings_schema(addtl_dict_entries: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if addtl_dict_entries is None:
        addtl_dict_entries = {}
    retval = {"type": ["array", "string"], "items": {"type": "string"}}
    retval.update(addtl_dict_entries)
    return retval


def make_object_or_array_of_objects_schema(single_object_schema: Dict[str, Any]) -> Dict[str, Any]:
    retval = {
        "oneOf": [
            {
                "title": "single_object_version",
                **single_object_schema,
            },
            {"title": "array_of_objects_version", "type": "array", "items": single_object_schema},
        ]
    }
    return retval


def with_attributes(schema: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    retval = {**schema, **kwargs}
    return retval
