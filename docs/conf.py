# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import sphinx_gallery
import sphinx_rtd_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "qolmat"
copyright = "2022, Quantmetry"
author = "Quantmetry"

# The full version, including alpha/beta/rc tags
version = "0.0.12"
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_markdown_tables",
    "sphinx_gallery.gen_gallery",
]
mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

from distutils.version import LooseVersion

# pngmath / imgmath compatibility layer for different sphinx versions
import sphinx

if LooseVersion(sphinx.__version__) < LooseVersion("1.4"):
    extensions.append("sphinx.ext.pngmath")
else:
    extensions.append("sphinx.ext.imgmath")

autodoc_default_flags = ["members", "inherited-members"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"
pygments_style = "friendly"

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Generate the plots for the gallery
plot_gallery = True

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx configuration
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/{.major}".format(sys.version_info),
        None,
    ),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
}

# sphinx-gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": ["../examples/tutorials/"],
    "gallery_dirs": ["examples/tutorials/"],
    "doc_module": "qolmat",
    "backreferences_dir": os.path.join("generated"),
    "reference_url": {"qolmat": None},
}


def setup(app):
    # a copy button to copy snippet of code from the documentation
    app.add_js_file("js/copybutton.js")
