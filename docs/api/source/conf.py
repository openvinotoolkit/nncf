# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
import os
import pkgutil
import sys
from typing import Any, Dict

from sphinx.ext.autodoc import mock

sys.path.insert(0, os.path.abspath("../../.."))

project = "NNCF"
html_title = "NNCF"
copyright_ = "2024, Intel Corporation"
author = "Intel Corporation"

extensions = ["autoapi.extension", "sphinx.ext.autodoc", "sphinx.ext.linkcode"]

# The below line in conjunction with specifying the 'sphinx.ext.autodoc' as extension
# makes the type hints from the function signature disappear from the signature in the HTML and instead
# show up in the function's documentation body. We use this to make the argument types in the documentation
# that are NNCF-defined types refer to the canonical API names instead of the actual fully-qualified names as defined by
# the position of the corresponding type - this is done by overriding the documented type directly in the docstring
# using :type: syntax.
autodoc_typehints = "description"

autoapi_dirs = ["../../../nncf"]
autoapi_options = ["members", "show-inheritance", "show-module-summary", "special-members", "imported-members"]

autoapi_template_dir = "_autoapi_templates"
autoapi_keep_files = True
autoapi_add_toctree_entry = False

html_theme_options = {
    "navigation_depth": -1,
}

exclude_patterns = []


class APIInfo:
    def __init__(self):
        self.api_names_vs_obj_dict: Dict[str, Any] = {}
        self.fqn_vs_canonical_name: Dict[str, str] = {}
        self.canonical_name_vs_fqn: Dict[str, str] = {}


def collect_api_entities() -> APIInfo:
    """
    Collects the fully qualified names of symbols in NNCF package that contain a special attribute (set via
    `nncf.common.api_marker.api` decorator) marking them as API entities.

    :return: A struct with API information, such as fully qualified names of API symbols and canonical name matching
      information.
    """
    retval = APIInfo()
    modules = {}
    skipped_modules: Dict[str, str] = {}
    import nncf

    for _, modname, _ in pkgutil.walk_packages(path=nncf.__path__, prefix=nncf.__name__ + ".", onerror=lambda x: None):
        try:
            modules[modname] = importlib.import_module(modname)
        except Exception as e:
            skipped_modules[modname] = str(e)

    from nncf.common.utils.api_marker import api

    canonical_imports_seen = set()

    for modname, module in modules.items():
        print(f"{modname}")
        for obj_name, obj in inspect.getmembers(module):
            objects_module = getattr(obj, "__module__", None)
            if (
                objects_module == modname
                and (inspect.isclass(obj) or inspect.isfunction(obj))
                and hasattr(obj, api.API_MARKER_ATTR)
            ):
                marked_object_name = obj._nncf_api_marker
                # Check the actual name of the originally marked object
                # so that the classes derived from base API classes don't
                # all automatically end up in API
                if marked_object_name != obj.__name__:
                    continue
                fqn = f"{modname}.{obj_name}"
                if hasattr(obj, api.CANONICAL_ALIAS_ATTR):
                    canonical_import_name = getattr(obj, api.CANONICAL_ALIAS_ATTR)
                    if canonical_import_name in canonical_imports_seen:
                        assert False, f"Duplicate canonical_alias detected: {canonical_import_name}"
                    retval.fqn_vs_canonical_name[fqn] = canonical_import_name
                    retval.canonical_name_vs_fqn[canonical_import_name] = fqn
                    canonical_imports_seen.add(canonical_import_name)
                    if canonical_import_name == fqn:
                        print(f"\t{obj_name}")
                    else:
                        print(f"\t{obj_name} -> {canonical_import_name}")
                retval.api_names_vs_obj_dict[fqn] = obj

    print()
    skipped_str = "\n".join([f"{k}: {v}" for k, v in skipped_modules.items()])
    print(f"Skipped: {skipped_str}\n")
    for fqn, canonical_alias in retval.fqn_vs_canonical_name.items():
        try:
            module_name, _, function_name = canonical_alias.rpartition(".")
            getattr(importlib.import_module(module_name), function_name)
        except (ImportError, AttributeError) as e:
            print(
                f"API entity with canonical_alias={canonical_alias} not available for import as specified!\n"
                f"Adjust the __init__.py files so that the symbol is available for import as {canonical_alias}."
            )
            raise e
        retval.api_names_vs_obj_dict[canonical_alias] = retval.api_names_vs_obj_dict.pop(fqn)

    print("API entities:")
    for api_fqn in retval.api_names_vs_obj_dict:
        print(api_fqn)
    return retval


mock_modules = [
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "openvino",
    "tensorflow",
    "keras",
    "tensorflow_addons",
    # Need add backend implementation functions to avoid endless loops on registered functions by mock module,
    "nncf.tensor.functions.numpy_numeric",
    "nncf.tensor.functions.numpy_linalg",
    "nncf.tensor.functions.torch_numeric",
    "nncf.tensor.functions.torch_linalg",
    "nncf.tensor.functions.ov",
]

with mock(mock_modules):
    api_info = collect_api_entities()

module_fqns = set()

for fqn_ in api_info.api_names_vs_obj_dict:
    path_elements = fqn_.split(".")
    for i in range(1, len(path_elements)):
        intermediate_module_path = ".".join(path_elements[:i])
        module_fqns.add(intermediate_module_path)


def skip_non_api(app, what, name, obj, skip, options):
    # AutoAPI-allowed callback to skip certain elements from generated documentation.
    # We use it to only allow API entities in the documentation (otherwise AutoAPI would generate docs for every
    # non-private symbol available in NNCF)
    if what in ["module", "package"] and name in module_fqns:
        print(f"skip_non_api: keeping module {name}")
        return False
    if what in ["method", "attribute", "property"]:
        class_name = name.rpartition(".")[0]
        if class_name in api_info.api_names_vs_obj_dict:
            return skip
    if name not in api_info.api_names_vs_obj_dict:
        skip = True
    else:
        print(f"skip_non_api: keeping API entity {name}")
        return False
    return skip


def linkcode_resolve(domain, info):
    # sphinx.ext.linkcode interface; will link to Github here.
    target_ref = "develop"
    base_url = f"https://github.com/openvinotoolkit/nncf/blob/{target_ref}/"
    if not info["module"]:
        return None
    fullname = info["module"] + "." + info["fullname"]

    if fullname not in api_info.api_names_vs_obj_dict:
        # Got a method/property description, info["fullname"] contained class.method_name
        fullname = fullname.rpartition(".")[0]

    if fullname in api_info.canonical_name_vs_fqn:
        fullname = api_info.canonical_name_vs_fqn[fullname]
        module_name = fullname.rpartition(".")[0]
    else:
        module_name = info["module"]
    filename = module_name.replace(".", "/")
    complete_url = base_url + filename + ".py"
    return complete_url


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_non_api)


html_show_sphinx = False
html_theme = "furo"
html_static_path = ["_static"]
