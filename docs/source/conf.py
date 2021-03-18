# type: ignore
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from typing import Any, List

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "FairScale"
copyright = "2020-2021, Facebook AI Research"
author = "Facebook AI Research"

# The full version, including alpha/beta/rc tags
release = "0.3.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # support NumPy and Google style docstrings
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[Any] = []


# -- Options for HTML output -------------------------------------------------


html_theme = "pytorch_sphinx_theme"
templates_path = ["_templates"]

# Add any paths that contain custom static files (such as style sheets) here,
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "includehidden": True,
    "canonical_url": "https://fairscale.readthedocs.io",
    "pytorch_project": "docs",
}

# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Over-ride PyTorch Sphinx css
def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {"url_resolver": lambda url: github_doc_root + url, "auto_toc_tree_section": "Contents"},
        True,
    )
    app.add_transform(AutoStructify)
    app.add_css_file("css/customize.css")
