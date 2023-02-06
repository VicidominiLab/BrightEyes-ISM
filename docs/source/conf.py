# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../../src/brighteyes_ism'))
sys.path.insert(0, os.path.abspath('../../src/brighteyes_ism'))
sys.path.insert(0, os.path.abspath('../..'))
  

project = 'BrightEyes-ISM'
copyright = '2023, A. Zunino'
author = 'A. Zunino'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = []


add_module_names = False
toc_object_entries_show_parents = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = ["_themes", ]
html_static_path = ['_static']
