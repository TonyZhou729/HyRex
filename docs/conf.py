# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('..'))

project = 'HyRex'
copyright = '2025, Zilu Zhou, Cara Giovanetti, and Hongwan Liu'
author = 'Zilu Zhou, Cara Giovanetti, and Hongwan Liu'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",   # Core extension for docstring extraction
    "sphinx.ext.napoleon",  # For Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Adds links to highlighted source code
    "myst_parser",          # (optional) Markdown support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
