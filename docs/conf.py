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
import hyrex as _h
sys.modules['HyRex'] = _h

project = 'HyRex'
copyright = '2025, Zilu Zhou, Cara Giovanetti, and Hongwan Liu'
author = 'Zilu Zhou, Cara Giovanetti, and Hongwan Liu'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


import re
import sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

def format_method_summaries(app, what, name, obj, options, lines):
    """
    Rewrite blocks like

        Recombination Unrelated Methods:
        rho_tot : Compute total energy density ...
        P_tot : Compute total pressure ...

    into valid reST bullet lists, so Sphinx preserves line breaks.
    """
    out = []
    in_methods_block = False
    just_opened_block = False

    for line in lines:
        # Detect headings that end with "Methods:"
        if re.match(r'^\s*.*Methods:\s*$', line):
            in_methods_block = True
            just_opened_block = True

            # keep the header line exactly as-is
            out.append(line)

            # IMPORTANT: add a blank line so the next lines
            # can become a proper bullet list in reST
            out.append("")
            continue

        # Blank line: close the block
        if line.strip() == "":
            in_methods_block = False
            just_opened_block = False
            out.append(line)
            continue

        if in_methods_block:
            # Look for "name : description ..."
            m = re.match(r'^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$', line)
            if m:
                ident, desc = m.groups()
                out.append(f'* ``{ident}``  {desc}')
                continue

        # default passthrough
        out.append(line)

    # mutate the list in-place so autodoc uses our version
    lines[:] = out

    # optional debug: shows up during sphinx-build
    logger.debug(f"[format_method_summaries] processed {name} ({what})")


def setup(app):
    # connect our transformer so it runs on every autodoc'd object
    app.connect('autodoc-process-docstring', format_method_summaries)

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
