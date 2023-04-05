# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib
import inspect
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import pkgutil
import sys
from typing import List

import nncf

sys.path.insert(0, os.path.abspath('../../..'))

project = 'nncf'
copyright = '2023, Intel Corporation'
author = 'Intel Corporation'
release = 'v2.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ['autoapi.extension']

autoapi_dirs = ['../../../nncf']
autoapi_options = ['members', 'show-inheritance',
                   'show-module-summary', 'special-members', 'imported-members']

autoapi_template_dir = '_autoapi_templates'
exclude_patterns = []


def collect_api_entities() -> List[str]:
    modules = {}
    skipped_modules = []
    for importer, modname, ispkg in pkgutil.walk_packages(path=nncf.__path__,
                                                          prefix=nncf.__name__+'.',
                                                          onerror=lambda x: None):
        try:
            modules[modname] = importlib.import_module(modname)
        except:
            skipped_modules.append(modname)

    api_fqns = []
    for modname, module in modules.items():
        print(f"{modname}")
        for obj_name, obj in inspect.getmembers(module):
            objects_module = getattr(obj, '__module__', None)
            if objects_module == modname:
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    if hasattr(obj, "_nncf_api_marker"):
                        print(f"\t{obj_name}")
                        api_fqns.append(f"{modname}.{obj_name}")

    print()
    skipped_str = '\n'.join(skipped_modules)
    print(f"Skipped: {skipped_str}\n")

    print("API entities:")
    for api_fqn in api_fqns:
        print(api_fqn)
    return api_fqns

api_fqns = collect_api_entities()


module_fqns = set()

for fqn in api_fqns:
    path_elements = fqn.split('.')
    for i in range(1, len(path_elements)):
        intermediate_module_path = '.'.join(path_elements[:i])
        module_fqns.add(intermediate_module_path)


def skip_non_api(app, what, name, obj, skip, options):
    if what in ["module", "package"] and name in module_fqns:
        print(f"VSHAMPOR: keeping module {name}")
        return skip
    if what in ["method", "attribute"]:
        class_name = name.rpartition('.')[0]
        if class_name in api_fqns:
            return skip
    if name not in api_fqns:
       skip = True
    else:
        print(f"VSHAMPOR: keeping API entity {name}")
    return skip


def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_non_api)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
