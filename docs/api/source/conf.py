# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))

project = 'nncf'
copyright = '2023, Intel Corporation'
author = 'Intel Corporation'
release = 'v2.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary']
autosummary_generate = True  # Turn on sphinx.ext.autosummary

templates_path = ['_templates']
exclude_patterns = []

module_fqn_with_api_fns_memo = set()

_memo = {}

def _has_api_members(module, memo):
    if module in memo:
        return memo[module]
    modules = []
    funcs_and_classes = []
    has_api = False
    for obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            funcs_and_classes.append(obj)
        if inspect.ismodule(obj):
            modules.append(obj)
    for fc in funcs_and_classes:
        if hasattr(fc, "_nncf_api_marker"):
            has_api = True
            break
    for submodule in modules:
        if _has_api_members(submodule, memo):
            has_api = True
            break
    memo[module] = has_api
    return has_api

def skip_non_api(app, what, name, obj, skip, options):
    if what == "module" and _has_api_members(obj, _memo):
        print(f"VSHAMPOR: not skipping module {name}, has API")
        return False
    elif hasattr(obj, "_nncf_api_marker"):
        print(f"VSHAMPOR: not skipping object {name}, is API")
        return False

    print(f"VSHAMPOR: skipping {what} {name}, not API")
    return True

def setup(sphinx):
   sphinx.connect("autodoc-skip-member", skip_non_api)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
