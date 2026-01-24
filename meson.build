project('slicot', 'c',
  version: '1.0.0',
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

if cc.has_function('dgemm_', dependencies: blas)
  add_project_arguments('-DSLC_FC_LOWER_US', language: 'c')
elif cc.has_function('dgemm', dependencies: blas)
  add_project_arguments('-DSLC_FC_LOWER', language: 'c')
else
  add_project_arguments('-DSLC_FC_LOWER_US', language: 'c')
endif

inc = include_directories('include')
subdir('src')
subdir('python')
