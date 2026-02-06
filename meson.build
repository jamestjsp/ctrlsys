project('slicot_c', 'c',
  version: '1.0.13',
  license: 'BSD-3-Clause',
  meson_version: '>=1.1.0',
  default_options: [
    'c_std=c11',
    'buildtype=debugoptimized',
    'warning_level=2',
  ],
)

cc = meson.get_compiler('c')

# BLAS/LAPACK dependency with fallback chain
blas_opt = get_option('blas')
lapack_opt = get_option('lapack')

if blas_opt == 'auto'
  # Try multiple BLAS implementations in order
  blas_dep = dependency('openblas', required: false)
  if not blas_dep.found()
    blas_dep = dependency('scipy-openblas', required: false)
  endif
  if not blas_dep.found()
    blas_dep = dependency('blas', required: false)
  endif
  if not blas_dep.found() and host_machine.system() == 'darwin'
    blas_dep = dependency('Accelerate', required: false)
  endif
  if not blas_dep.found()
    # Final fallback: try to find BLAS library directly
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
    # Check if BLAS bundles LAPACK (e.g., OpenBLAS) BEFORE trying system liblapack.
    # On musllinux, system liblapack depends on system openblas which conflicts
    # with scipy-openblas32 at link time. Test all naming conventions since
    # convention detection hasn't run yet.
    if cc.has_function('dgesv_', dependencies: [blas_dep]) or cc.has_function('dgesv', dependencies: [blas_dep]) or cc.has_function('DGESV', dependencies: [blas_dep])
      lapack_dep = blas_dep
      message('LAPACK bundled in BLAS library')
    else
      lapack_dep = cc.find_library('lapack', required: true)
    endif
  endif
else
  lapack_dep = dependency(lapack_opt, required: true)
endif

# Detect BLAS/LAPACK symbol naming convention
blas_lapack_deps = [blas_dep, lapack_dep]

# Try lowercase with underscore (most common: gfortran)
if cc.has_function('dgemm_', dependencies: blas_lapack_deps)
  fc_defines = ['-DSLC_FC_LOWER_US']
  message('BLAS/LAPACK uses lowercase with underscore: dgemm_')
# Try lowercase only (some compilers)
elif cc.has_function('dgemm', dependencies: blas_lapack_deps)
  fc_defines = ['-DSLC_FC_LOWER']
  message('BLAS/LAPACK uses lowercase: dgemm')
# Try uppercase (old style)
elif cc.has_function('DGEMM', dependencies: blas_lapack_deps)
  fc_defines = ['-DSLC_FC_UPPER']
  message('BLAS/LAPACK uses uppercase: DGEMM')
else
  # Default to lowercase with underscore
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
