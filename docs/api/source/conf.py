import importlib
import inspect
import os
import pkgutil
import sys
from typing import Dict, List

from sphinx.ext.autodoc import mock

sys.path.insert(0, os.path.abspath("../../.."))

project = "nncf"
copyright = "2023, Intel Corporation"
author = "Intel Corporation"
release = "v2.4.0"

extensions = ["autoapi.extension", "sphinx.ext.autodoc"]
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


def collect_api_entities() -> List[str]:
    """
    Collects the fully qualified names of symbols in NNCF package that contain a special attribute (set via
    `nncf.common.api_marker.api` decorator) marking them as API entities.
    :return: A list of fully qualified names of API symbols.
    """
    modules = {}
    skipped_modules = {}  # type: Dict[str, str]
    import nncf

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=nncf.__path__, prefix=nncf.__name__ + ".", onerror=lambda x: None
    ):
        try:
            modules[modname] = importlib.import_module(modname)
        except Exception as e:
            skipped_modules[modname] = str(e)

    from nncf.common.utils.api_marker import api

    api_fqns = dict()
    aliased_fqns = {}  # type: Dict[str, str]
    canonical_imports_seen = set()
    for modname, module in modules.items():
        print(f"{modname}")
        for obj_name, obj in inspect.getmembers(module):
            objects_module = getattr(obj, "__module__", None)
            if objects_module == modname:
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    if hasattr(obj, api.API_MARKER_ATTR):
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
                            aliased_fqns[fqn] = canonical_import_name
                            canonical_imports_seen.add(canonical_import_name)
                            if canonical_import_name == fqn:
                                print(f"\t{obj_name}")
                            else:
                                print(f"\t{obj_name} -> {canonical_import_name}")
                        api_fqns[fqn] = obj

    print()
    skipped_str = "\n".join([f"{k}: {v}" for k, v in skipped_modules.items()])
    print(f"Skipped: {skipped_str}\n")
    for fqn, canonical_alias in aliased_fqns.items():
        try:
            module_name, _, function_name = canonical_alias.rpartition(".")
            getattr(importlib.import_module(module_name), function_name)
        except (ImportError, AttributeError) as e:
            print(
                f"API entity with canonical_alias={canonical_alias} not available for import as specified!\n"
                f"Adjust the __init__.py files so that the symbol is available for import as {canonical_alias}."
            )
            raise e
        api_fqns[canonical_alias] = api_fqns.pop(fqn)

    print("API entities:")
    for api_fqn in api_fqns:
        print(api_fqn)
    return list(api_fqns.keys())


with mock(["torch", "torchvision", "onnx", "onnxruntime", "openvino", "tensorflow", "tensorflow_addons"]):
    api_fqns = collect_api_entities()

module_fqns = set()

for fqn in api_fqns:
    path_elements = fqn.split(".")
    for i in range(1, len(path_elements)):
        intermediate_module_path = ".".join(path_elements[:i])
        module_fqns.add(intermediate_module_path)


def skip_non_api(app, what, name, obj, skip, options):
    # AutoAPI-allowed callback to skip certain elements from generated documentation.
    # We use it to only allow API entities in the documentation (otherwise AutoAPI would generate docs for every
    # non-private symbol available in NNCF)
    if what in ["module", "package"] and name in module_fqns:
        print(f"skip_non_api: keeping module {name}")
        return skip
    if what in ["method", "attribute"]:
        class_name = name.rpartition(".")[0]
        if class_name in api_fqns:
            return skip
    if name not in api_fqns:
        skip = True
    else:
        print(f"skip_non_api: keeping API entity {name}")
        return False
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_non_api)


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
