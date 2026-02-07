project('slicot_c', 'c',
  version: '1.0.14',
  license: 'BSD-3-Clause',
  meson_version: '>=1.1.0',
  default_options: [
    'c_std=c11',
    'buildtype=debugoptimized',
    'warning_level=2',
  ],
)

cc = meson.get_compiler('c')

warning_flags = ['-Wno-unused-parameter']
if cc.has_argument('-Wno-cast-function-type-mismatch')
  warning_flags += ['-Wno-cast-function-type-mismatch']
elif cc.has_argument('-Wno-cast-function-type')
  warning_flags += ['-Wno-cast-function-type']
endif
add_project_arguments(warning_flags, language: 'c')

# BLAS/LAPACK dependency with fallback chain
blas_opt = get_option('blas')
lapack_opt = get_option('lapack')

blas_is_openblas = false

if blas_opt == 'auto'
  blas_dep = dependency('openblas', required: false)
  if blas_dep.found()
    blas_is_openblas = true
  endif
  if not blas_dep.found()
    blas_dep = dependency('scipy-openblas', required: false)
    if blas_dep.found()
      blas_is_openblas = true
    endif
  endif
  if not blas_dep.found()
    blas_dep = dependency('blas', required: false)
  endif
  if not blas_dep.found() and host_machine.system() == 'darwin'
    blas_dep = dependency('Accelerate', required: false)
  endif
  if not blas_dep.found()
    blas_dep = cc.find_library('blas', required: true)
  endif
else
  blas_dep = dependency(blas_opt, required: true)
endif

if lapack_opt == 'auto'
  lapack_dep = dependency('lapack', required: false)
  if not lapack_dep.found() and host_machine.system() == 'darwin'
    lapack_dep = dependency('Accelerate', required: false)
  endif
  if not lapack_dep.found()
    if blas_is_openblas
      lapack_dep = blas_dep
      message('LAPACK bundled in OpenBLAS')
    else
      lapack_dep = cc.find_library('lapack', required: true)
    endif
  endif
else
  lapack_dep = dependency(lapack_opt, required: true)
endif

# Detect BLAS/LAPACK symbol naming convention
blas_lapack_deps = [blas_dep, lapack_dep]

if cc.has_function('scipy_dgemm_', dependencies: blas_lapack_deps)
  fc_defines = ['-DSLC_FC_SCIPY_OPENBLAS']
  message('BLAS/LAPACK uses scipy-openblas32 prefix: scipy_dgemm_')
elif cc.has_function('dgemm_', dependencies: blas_lapack_deps)
  fc_defines = ['-DSLC_FC_LOWER_US']
  message('BLAS/LAPACK uses lowercase with underscore: dgemm_')
elif cc.has_function('dgemm', dependencies: blas_lapack_deps)
  fc_defines = ['-DSLC_FC_LOWER']
  message('BLAS/LAPACK uses lowercase: dgemm')
elif cc.has_function('DGEMM', dependencies: blas_lapack_deps)
  fc_defines = ['-DSLC_FC_UPPER']
  message('BLAS/LAPACK uses uppercase: DGEMM')
else
  fc_defines = ['-DSLC_FC_LOWER_US']
  warning('Could not detect BLAS/LAPACK naming convention, defaulting to dgemm_')
endif

# ILP64 support
if get_option('ilp64')
  fc_defines += ['-DSLICOT_ILP64']
  message('Using 64-bit integers (ILP64)')
endif

# Version defines - derived from project version
version_parts = meson.project_version().split('.')
fc_defines += [
  '-DSLICOT_VERSION_MAJOR=' + version_parts[0],
  '-DSLICOT_VERSION_MINOR=' + version_parts[1],
  '-DSLICOT_VERSION_PATCH=' + version_parts[2],
]

# Generate slicot_config.h from template
conf_data = configuration_data()
conf_data.set('SLICOT_VERSION_MAJOR', version_parts[0])
conf_data.set('SLICOT_VERSION_MINOR', version_parts[1])
conf_data.set('SLICOT_VERSION_PATCH', version_parts[2])
configure_file(
  input: 'include/slicot_config.h.in',
  output: 'slicot_config.h',
  configuration: conf_data,
  install_dir: get_option('includedir') / 'slicot',
)

# Include directories (build root for generated slicot_config.h, include/ for rest)
inc = include_directories('include', '.')

# Build the library
subdir('src')

# Python bindings
subdir('python')

# Benchmarks
subdir('benchmarks')
