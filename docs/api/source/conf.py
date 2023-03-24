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

def skip_non_api(app, what, name, obj, skip, options):
    if what == "module":
        objs = inspect.getmembers(obj)
    if hasattr(obj, "_nncf_api_marker"):
        return False
    return True

def setup(sphinx):
   sphinx.connect("autodoc-skip-member", skip_non_api)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
