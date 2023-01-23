"""
 Copyright (c) 2023 Intel Corporation
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

from copy import deepcopy
from pathlib import Path
from typing import List
from typing import Optional
from typing import Type

import jsonschema
import jstyleson as json

from nncf.common.logging import nncf_logger
from nncf.common.utils.os import safe_open
from nncf.config.definitions import SCHEMA_VISUALIZATION_URL
from nncf.config.schema import REF_VS_ALGO_SCHEMA
from nncf.config.schema import NNCF_CONFIG_SCHEMA
from nncf.config.schema import validate_single_compression_algo_schema
from nncf.config.structures import NNCFExtraConfigStruct


class NNCFConfig(dict):
    """A regular dictionary object extended with some utility functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__nncf_extra_structs = {}  # type: dict[str, NNCFExtraConfigStruct]

    @classmethod
    def from_dict(cls, nncf_dict):
        """
        Load NNCF config from dict;
        The dict must contain only json supported primitives.
        """

        NNCFConfig.validate(nncf_dict)
        return cls(deepcopy(nncf_dict))

    @classmethod
    def from_json(cls, path) -> 'NNCFConfig':
        file_path = Path(path).resolve()
        with safe_open(file_path) as f:
            loaded_json = json.load(f)
        return cls.from_dict(loaded_json)

    def register_extra_structs(self, struct_list: List[NNCFExtraConfigStruct]):
        for struct in struct_list:
            struct_id = struct.get_id()
            if struct_id in self.__nncf_extra_structs:
                raise RuntimeError(f'{struct_id} is already registered as extra struct in NNCFConfig!')
            self.__nncf_extra_structs[struct_id] = struct

    def get_extra_struct(self, struct_cls: Type[NNCFExtraConfigStruct]) -> NNCFExtraConfigStruct:
        return self.__nncf_extra_structs[struct_cls.get_id()]

    def has_extra_struct(self, struct_cls: Type[NNCFExtraConfigStruct]) -> NNCFExtraConfigStruct:
        return struct_cls.get_id() in self.__nncf_extra_structs

    def get_all_extra_structs_for_copy(self) -> List[NNCFExtraConfigStruct]:
        return list(self.__nncf_extra_structs.values())

    def get_redefinable_global_param_value_for_algo(self, param_name: str, algo_name: str) -> Optional:
        """
        Some parameters can be specified both on the global NNCF config .json level (so that they apply
        to all algos), and at the same time overridden in the algorithm-specific section of the .json.
        This function returns the value that should apply for a given algorithm name, considering the
        exact format of this config.

        :param param_name: The name of a parameter in the .json specification of the NNCFConfig, that may
          be present either at the top-most level of the .json, or at the top level of the algorithm-specific
          subdict.
        :param algo_name: The name of the algorithm (among the allowed algorithm names in the .json) for which
          the resolution of the redefinable parameter should occur.
        :return: The value of the parameter that should be applied for the algo specified by `algo_name`.
        """
        from nncf.config.extractors import extract_algo_specific_config #pylint: disable=cyclic-import
        algo_config = extract_algo_specific_config(self, algo_name)
        param = self.get(param_name)
        algo_specific_param = algo_config.get(param_name)
        if algo_specific_param is not None:
            param = algo_specific_param
        return param

    @staticmethod
    def schema():
        return NNCF_CONFIG_SCHEMA

    @staticmethod
    def _is_path_to_algorithm_name(path_parts: List[str]) -> bool:
        return (len(path_parts) == 2 and path_parts[0] == "compression" and path_parts[1] == "algorithm") or \
               (len(path_parts) == 3 and path_parts[0] == "compression" and path_parts[1].isnumeric()
                and path_parts[2] == "algorithm")

    @staticmethod
    def validate(loaded_json):
        try:
            jsonschema.validate(loaded_json, NNCFConfig.schema())
        except jsonschema.ValidationError as e:
            nncf_logger.error('Invalid NNCF config supplied!')
            absolute_path_parts = [str(x) for x in e.absolute_path]
            if not NNCFConfig._is_path_to_algorithm_name(absolute_path_parts):
                e.message += f"\nRefer to the NNCF config schema documentation at " \
                             f"{SCHEMA_VISUALIZATION_URL}"
                e.schema = "*schema too long for stdout display*"
                raise e

            # Need to make the error more algo-specific in case the config was so bad that no
            # scheme could be matched
            # If error is in the algo section, will revalidate the algo sections separately to
            # make the error message more targeted instead of displaying the entire huge schema.
            compression_section = loaded_json["compression"]
            if isinstance(compression_section, dict):
                validate_single_compression_algo_schema(compression_section, REF_VS_ALGO_SCHEMA)
            else:
                # Passed a list of dicts
                for compression_algo_dict in compression_section:
                    validate_single_compression_algo_schema(compression_algo_dict, REF_VS_ALGO_SCHEMA)
