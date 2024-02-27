# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "SCALib"
copyright = "SIMPLE Contributors"


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

bibtex_bibfiles = ["refs.bib"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

autoclass_content = "both"

autosummary_ignore_module_all = False

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "special-members": False,
}

autodoc_typehints = "description"

# We do not need to install an build the package to generate the doc.
# We only have to mock dependencies, binary packages, and add the source to the
# python path.
autodoc_mock_imports = [
    "scalib._scalib_ext",
    "scalib.version",
    "scalib.build_config",
    "numpy",
    "cpuinfo",
]

user_agent = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0"



import sys

sys.path.append("../src")

# from scalib.version import version
# The full version, including alpha/beta/rc tags
# release = version
