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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os.path

# -- Project information -----------------------------------------------------

project = 'SPU'
copyright = '2021 Ant Group Co., Ltd.'
author = 'SPU authors'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
    "nbsphinx",
    'sphinxcontrib.actdiag',
    'sphinxcontrib.blockdiag',
    'sphinxcontrib.mermaid',
    'sphinxcontrib.nwdiag',
    'sphinxcontrib.packetdiag',
    'sphinxcontrib.rackdiag',
    'sphinxcontrib.seqdiag',
    'sphinx_markdown_tables',
]

nbsphinx_requirejs_path = ''

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
# html_theme_options = {'page_width': 'max-content'}
# html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# note: 'bysource' does not work for c++ extensions
autodoc_member_order = 'groupwise'

# Enable TODO
todo_include_todos = True

# config blockdiag

# global variables
extlinks = {
    'spu_doc_host': ('https://spu.readthedocs.io/zh/latest', 'doc '),
    'spu_code_host': ('https://github.com/secretflow', 'code '),
}

font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
if os.path.isfile(font_file):
    blockdiag_fontpath = font_file
    seqdiag_fontpath = font_file


html_favicon = '_static/favicon.ico'

html_css_files = [
    'css/custom.css',
]

html_js_files = ['js/custom.js']

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/secretflow/spu",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "external_links": [
        {"name": "SecretFlow", "url": "https://secretflow.readthedocs.io/"},
    ],
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

suppress_warnings = ["myst.header"]

# myst_heading_anchors = 3
# myst_commonmark_only = True
myst_gfm_only = True
myst_heading_anchors = 1
myst_title_to_header = True

# app setup hook
def setup(app):
    app.add_config_value(
        'recommonmark_config',
        {
            # 'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
        },
        True,
    )
