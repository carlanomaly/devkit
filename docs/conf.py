"""Sphinx configuration for the CarlAnomaly devkit."""

import html
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

project = "CarlAnomaly DevKit"
author = "Konstantin Kirchheim"
copyright = "2026"
release = "0.1.0"
version = "0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------

autodoc_member_order = "bysource"
# No "Bases:" line, as the base classes are private. Members are listed per
# directive (:members:), so dataset pages show only the class docstrings.
autodoc_default_options = {
    "undoc-members": False,
}

# ---------------------------------------------------------------------------
# napoleon (NumPy-style docstrings)
# ---------------------------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# sphinx_autodoc_typehints adds the annotated types to the Parameters list that
# napoleon renders (napoleon_use_param), so each parameter appears once. Return
# values are described in the docstrings; the annotations (Dict[str, Any], Tensor)
# would add nothing.
typehints_document_rtype = False

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

# Furo, restyled after carlanomaly.de: _templates/page.html adds the site header
# and footer, _static/custom.css the website's fonts, colours and syntax colours.
html_theme = "furo"
html_title = "CarlAnomaly DevKit"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_js_files = [("site-header.js", {"defer": "defer"}), ("search.js", {"defer": "defer"})]

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


# ---------------------------------------------------------------------------
# Search (Pagefind)
# ---------------------------------------------------------------------------
# After an HTML build, Pagefind indexes the pages' main content (marked with
# data-pagefind-body in _templates/page.html) into _build/html/pagefind/;
# _static/search.js queries it as the reader types. Without the index the
# header form falls back to Sphinx's own search page.

from sphinx.writers.html5 import HTML5Translator  # noqa: E402


class _SearchTranslator(HTML5Translator):
    """HTML output adjusted for the Pagefind index.

    * Each section's id is rendered on its heading instead of on
      ``<section>``: Pagefind links results to headings that carry an id.
      Anchors and permalinks stay the same.
    * Every API signature indexes its name as separate words. Sphinx renders
      ``carlanomaly.require_parts(`` without spaces, which Pagefind reads as
      one word, so a search for ``require_parts`` would not find it.
    """

    def visit_section(self, node) -> None:
        ids, node["ids"] = node["ids"], []
        super().visit_section(node)
        node["ids"] = ids

    def section_title_tags(self, node):
        own = node["ids"]
        node["ids"] = own + node.parent["ids"]
        try:
            return super().section_title_tags(node)
        finally:
            node["ids"] = own

    def visit_desc_signature(self, node) -> None:
        super().visit_desc_signature(node)
        name = node.get("fullname")
        if name:
            words = " ".join(dict.fromkeys(name.split(".") + [name]))
            self.body.append(
                f'<span data-pagefind-index-attrs="data-name" data-name="{html.escape(words)}"></span>'
            )


def _pagefind(app, exception) -> None:
    if exception is not None or app.builder.format != "html":
        return
    from sphinx.util import logging

    logger = logging.getLogger(__name__)
    cmd = [
        sys.executable, "-m", "pagefind",
        "--site", str(app.outdir),
        "--exclude-selectors", ".headerlink, .viewcode-link",
        "--include-characters", "_",
        "--quiet",
    ]
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        logger.warning(f"pagefind failed, search falls back to Sphinx's page ({err}); "
                       "install the docs extra: pip install -e '.[docs]'")


# The dataset pages give each class a short signature by hand (e.g.
# ``RGBDataset(root, split="train", direction="front", **kwargs)``) instead of
# the real one, which lists every forwarded argument. sphinx_autodoc_typehints
# would replace it with the real signature again, so this handler, which runs
# first (lower priority number), keeps the signature as given.
def _keep_dataset_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class" and name.startswith("carlanomaly.datasets.") and signature:
        return signature, return_annotation
    return None


def setup(app):
    app.connect("autodoc-process-signature", _keep_dataset_signature, priority=400)
    app.set_translator("html", _SearchTranslator)
    app.connect("build-finished", _pagefind)

