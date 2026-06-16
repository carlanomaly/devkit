"""Sphinx configuration for the CarlAnomaly devkit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

project = "CarlAnomaly DevKit"
author = "Konstantin Kirchheim"
copyright = "2024, Konstantin Kirchheim"
release = "0.1.0"
version = "0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__len__, __getitem__",
}

# ---------------------------------------------------------------------------
# napoleon (NumPy-style docstrings)
# ---------------------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_title = "CarlAnomaly DevKit"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/carlanomaly/devkit",
    "source_branch": "main",
    "source_directory": "docs/",
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
default_role = "py:obj"
