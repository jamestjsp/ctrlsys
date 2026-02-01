project('slicot', 'c',
  version: '1.0.8',
  license: 'BSD-3-Clause',
  meson_version: '>=1.1.0',
  default_options: ['c_std=c11', 'buildtype=release']
)

cc = meson.get_compiler('c')
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

blas_opt = get_option('blas')
lapack_opt = get_option('lapack')

if blas_opt == 'auto'
  blas = dependency('openblas', required: false)
  if not blas.found()
    blas = dependency('scipy-openblas', required: false)
  endif
  if not blas.found()
    blas = dependency('blas', required: false)
  endif
  if not blas.found()
    _openblas_info = run_command(py, '-c',
      'import scipy_openblas32 as sc; print(sc.get_include_dir()); print(sc.get_lib_dir())',
      check: false)
    if _openblas_info.returncode() == 0
      _lines = _openblas_info.stdout().strip().split('\n')
      blas = declare_dependency(
        include_directories: include_directories(_lines[0]),
        dependencies: cc.find_library('scipy_openblas', dirs: [_lines[1]])
      )
    endif
  endif
  if not blas.found()
    blas = dependency('Accelerate', required: true)
  endif
else
  blas = dependency(blas_opt, required: true)
endif

if lapack_opt == 'auto'
  lapack = dependency('lapack', required: false)
  if not lapack.found()
    lapack = blas
  endif
else
  lapack = dependency(lapack_opt, required: true)
endif

if cc.has_function('scipy_dgemm_', dependencies: blas)
  add_project_arguments('-DSLC_FC_SCIPY_OPENBLAS', language: 'c')
elif cc.has_function('dgemm_', dependencies: blas)
  add_project_arguments('-DSLC_FC_LOWER_US', language: 'c')
elif cc.has_function('dgemm', dependencies: blas)
  add_project_arguments('-DSLC_FC_LOWER', language: 'c')
else
  add_project_arguments('-DSLC_FC_LOWER_US', language: 'c')
endif

inc = include_directories('include')
subdir('src')
subdir('python')
subdir('benchmarks')
