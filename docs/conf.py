# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import tomllib
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src/marloes")) #FIXME: Since you guys don't use the standard file structure, it has some issues finding the modules
sys.path.insert(0, os.path.abspath(os.path.join(".", "_ext")))

with open("index.md") as fp:
    for i, line in enumerate(fp):
        if i == 12:
            project = line[:-1]

project = project.replace(" ", "").capitalize()
print("Building documentation for project: '" + project + "'")

copyright = str(datetime.today().year) + ", REpowered B.V"
author = "the Repowered team"
with open(os.path.join("..", "pyproject.toml"), "rb") as f:
    release = tomllib.load(f)["tool"]["poetry"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "_ext", "Thumbs.db", ".DS_Store"]
autodoc_mock_imports = []
autoclass_content = "class"
autodoc_member_order = "bysource"
autosummary_generate = True
myst_enable_extensions = ["substitution"]
myst_substitutions = {"Package": project}

# -- Pydantic autodoc configuration -----------------------------------------
show_json_representations = (
    False  # Whether to show json representation of models or settings
)
show_summaries = True  # Whether to show summaries under models/settings
elaborate_field_info = (
    False  # Whether to list validators and constraints under each field
)
member_ordering = "default"  # How to order summaries, models and settings
ordering = {
    "summary": {"default": "alphabetical", "bysource": "bysource"},
    "member": {"default": "groupwise", "bysource": "bysource"},
}
autodoc_pydantic_model_summary_list_order = ordering["summary"][
    member_ordering
]  # ┰─ Ordering
autodoc_pydantic_settings_summary_list_order = ordering["summary"][member_ordering]  # ┃
autodoc_pydantic_model_member_order = ordering["member"][member_ordering]  # ┃
autodoc_pydantic_settings_member_order = ordering["member"][member_ordering]  # ┚
autodoc_pydantic_model_show_json = show_json_representations  # ┰─ Json representations
autodoc_pydantic_settings_show_json = show_json_representations  # ┚
autodoc_pydantic_model_show_config_summary = show_summaries  # ┰─ Summaries
autodoc_pydantic_model_show_validator_summary = show_summaries  # ┃
autodoc_pydantic_model_show_field_summary = show_summaries  # ┃
autodoc_pydantic_settings_show_config_summary = show_summaries  # ┃
autodoc_pydantic_settings_show_validator_summary = show_summaries  # ┃
autodoc_pydantic_settings_show_field_summary = show_summaries  # ┚
autodoc_pydantic_field_list_validators = elaborate_field_info  # ┰─ Elaborate field info
autodoc_pydantic_field_show_constraints = elaborate_field_info  # ┚

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_favicon = "https://www.google.com/s2/favicons?domain=www.repowered.nl"
html_theme_options = {
    "logo": {
        "text": project + " v" + release + " Docs",
        "image_light": "https://www.repowered.nl/wp-content/build/svg/logo-repowered.svg",
        "image_dark": "https://www.repowered.nl/wp-content/build/svg/logo-repowered.svg",
        "alt_text": project + " v" + release + " Docs - Repowered Docs Home",
        "link": "https://docs.repowered.nl/",
    }
}
