# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'SCALib'
copyright = '2021, UCLouvain'


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.mathjax',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

autoclass_content = 'both'
numpydoc_show_class_members = False

# We do not need to install an build the package to generate the doc.
# We only have to mock dependencies, binary packages, and add the source to the
# python path.
autodoc_mock_imports = [
        "scalib._scalib_ext",
        "scalib.version",
        "numpy",
        ]

import sys
sys.path.append('../src')

#from scalib.version import version
# The full version, including alpha/beta/rc tags
#release = version

