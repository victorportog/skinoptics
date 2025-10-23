# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import sys
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('furo')
install('sphinxemoji')
install('skinoptics')
sys.path.insert(0, os.path.abspath('../..'))
import skinoptics

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SkinOptics'
copyright = '2024-2025, Victor Lima'
author = 'Victor Lima'
release = '0.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.doctest', 'sphinx.ext.autodoc', 'sphinxemoji.sphinxemoji']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
# svg file downloaded from https://openmoji.org/library/emoji-2728/
html_favicon = '_static/openmoji_1F506.svg'
html_title = 'SkinOptics documentation - version ' + release

html_extra_path = ['_static/tutorial_optical_properties.html', '_static/tutorial_colors.html',
'_static/crosscheck_colors.html', '_static/reproducing_2011DelgadoAtencio.html',
'_static/validation_Delta_E_00.html']
