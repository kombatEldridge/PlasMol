# Sphinx configuration for PlasMol documentation
# https://www.sphinx-doc.org/

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
project = "PlasMol"
author = "Brinton King Eldridge"
copyright = f"{datetime.now().year}, {author}"
release = "1.2.0"
version = "1.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "site",  # old mkdocs output
    "latex",
]

# MyST (Markdown)
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_dmath_double_inline = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_favicon = None
html_show_sourcelink = True
html_copy_source = True

# Root document
root_doc = "index"

# Quiet common MyST/markdown issues from existing content
suppress_warnings = [
    "myst.header",
    "misc.highlighting_failure",
]
