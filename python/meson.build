py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# NumPy dependency
np_dep = dependency('numpy', required: false)
if not np_dep.found()
  # Try to get numpy include path from Python
  numpy_include = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'],
    check: true).stdout().strip()
  np_dep = declare_dependency(compile_args: ['-I' + numpy_include])
endif

# Generate docstrings.h from docstrings.json
docstrings_gen = custom_target('docstrings',
  input: 'data/docstrings.json',
  output: 'docstrings.h',
  command: [py, meson.project_source_root() / 'tools/generate_docstrings.py'],
  depend_files: [meson.project_source_root() / 'tools/generate_docstrings.py'],
)

# Python wrapper sources
wrapper_sources = files(
  'slicot_module.c',
  'wrappers/py_ab.c',
  'wrappers/py_ag.c',
  'wrappers/py_bb.c',
  'wrappers/py_bd.c',
  'wrappers/py_de.c',
  'wrappers/py_df.c',
  'wrappers/py_dg.c',
  'wrappers/py_dk.c',
  'wrappers/py_fb.c',
  'wrappers/py_fd.c',
  'wrappers/py_ib.c',
  'wrappers/py_lapack_aux.c',
  'wrappers/py_ma.c',
  'wrappers/py_mb01.c',
  'wrappers/py_mb02.c',
  'wrappers/py_mb03.c',
  'wrappers/py_mb04.c',
  'wrappers/py_mb05.c',
  'wrappers/py_mc.c',
  'wrappers/py_md.c',
  'wrappers/py_nf.c',
  'wrappers/py_sb.c',
  'wrappers/py_sg.c',
  'wrappers/py_tb.c',
  'wrappers/py_tc.c',
  'wrappers/py_td.c',
  'wrappers/py_tf.c',
  'wrappers/py_tg.c',
  'wrappers/py_ud.c',
  'wrappers/py_ue.c',
)

# Build Python extension module
py.extension_module('_slicot',
  [wrapper_sources, docstrings_gen],
  include_directories: [inc, 'data'],
  dependencies: [py_dep, np_dep, slicot_dep],
  c_args: fc_defines,
  subdir: 'ctrlsys',
  install: true,
)

# Install Python package files
py.install_sources(
  '_package_init/__init__.py',
  subdir: 'ctrlsys',
)
