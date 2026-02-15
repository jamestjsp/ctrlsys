from importlib.metadata import version as get_version

project = 'ctrlsys'
copyright = '2024, James Joseph'
author = 'James Joseph'
release = get_version('ctrlsys')

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_llms_txt',
]

html_baseurl = 'https://ctrlsys.readthedocs.io/en/latest/'

llms_txt_title = 'ctrlsys - Control Theory Library for Python'
llms_txt_summary = (
    'Python bindings for SLICOT (Subroutine Library in Control Theory). '
    'C11 translation of the classic Fortran77 library with 578 numerical '
    'routines for systems and control. Install: pip install ctrlsys. '
    'All arrays must use Fortran column-major order (order="F"). '
    'Functions return tuples; last element is info (0=success, <0=bad arg, >0=algorithm error).'
)

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
