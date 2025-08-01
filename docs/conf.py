#!/usr/bin/env python3
import datetime
import os
import sys
from pathlib import Path

import pyvisgen

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
pyproject = tomllib.loads(pyproject_path.read_text())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_copybutton",
]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "changes",
    "*.log",
]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"


project = pyproject["project"]["name"]
author = pyproject["project"]["authors"][0]["name"]
copyright = "{}.  Last updated {}".format(
    author, datetime.datetime.now().strftime("%d %b %Y %H:%M")
)
python_requires = pyproject["project"]["requires-python"]

# make some variables available to each page
rst_epilog = f"""
.. |python_requires| replace:: {python_requires}
"""


version = pyvisgen.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- Version switcher -----------------------------------------------------

# Define the json_url for our version switcher.
json_url = "https://pyvisgen.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.,
version_match = os.getenv("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
if not version_match or version_match.isdigit():
    # For local development, infer the version to match from the package.
    if "dev" in release or "rc" in release:
        version_match = "latest"
    else:
        version_match = release

    # We want to keep the relative reference when on a pull request or locally
    json_url = "_static/switcher.json"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_file_suffix = ".html"

html_css_files = ["pyvisgen.css"]

html_favicon = "_static/favicon/favicon.ico"

html_theme_options = {
    "github_url": "https://github.com/radionets-project/pyvisgen",
    "header_links_before_dropdown": 5,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "version_match": version_match,
        "json_url": json_url,
    },
    "navigation_with_keys": False,
    "show_version_warning_banner": True,
    "icon_links_label": "Quick Links",
    "icon_links": [
        {
            "name": "Radionets Project",
            "url": "https://github.com/radionets-project",
            "type": "url",
            "icon": "https://avatars.githubusercontent.com/u/77392854?s=200&v=4",  # noqa: E501
        },
    ],
    "logo": {
        "image_light": "_static/pyvisgen.webp",
        "image_dark": "_static/pyvisgen_dark.webp",
        "alt_text": "pyvisgen",
    },
    "announcement": """
        <p>pyvisgen is not stable yet, so expect large and rapid
        changes to structure and functionality as we explore various
        design choices before the 1.0 release.</p>
    """,
}

html_title = f"{project}: Visibility Simulations in Python"
htmlhelp_basename = project + "docs"


# Configuration for intersphinx
intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


suppress_warnings = [
    "intersphinx.external",
]
