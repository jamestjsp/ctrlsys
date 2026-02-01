# Native C benchmark executables

bench_inc = include_directories('common')
bench_data_dir = meson.project_source_root() / 'SLICOT-Reference/benchmark_data'

# Linux needs _POSIX_C_SOURCE for clock_gettime and libm for sqrt
bench_args = []
if host_machine.system() == 'linux'
  bench_args += ['-D_POSIX_C_SOURCE=199309L']
endif
m_dep = cc.find_library('m', required: false)
bench_deps = [blas, lapack, m_dep]

# BB01AD benchmark - CAREX example generator
bench_bb01ad = executable('bench_bb01ad',
  'native/bench_bb01ad.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# BB02AD benchmark - DAREX example generator (discrete Riccati)
bench_bb02ad = executable('bench_bb02ad',
  'native/bench_bb02ad.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# BB03AD benchmark - CTLEX example generator (continuous Lyapunov)
bench_bb03ad = executable('bench_bb03ad',
  'native/bench_bb03ad.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# BB04AD benchmark - DTLEX example generator (discrete Lyapunov)
bench_bb04ad = executable('bench_bb04ad',
  'native/bench_bb04ad.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# BD01AD benchmark - CTDSX example generator
bench_bd01ad = executable('bench_bd01ad',
  'native/bench_bd01ad.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# BD02AD benchmark - DTDSX example generator (discrete descriptor systems)
bench_bd02ad = executable('bench_bd02ad',
  'native/bench_bd02ad.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# SB02MD benchmark - Continuous-time Algebraic Riccati Equation solver
bench_sb02md = executable('bench_sb02md',
  'native/bench_sb02md.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# SB02OD benchmark - Discrete-time Algebraic Riccati Equation solver
bench_sb02od = executable('bench_sb02od',
  'native/bench_sb02od.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# SB03MD benchmark - Continuous-time Lyapunov solver
bench_sb03md = executable('bench_sb03md',
  'native/bench_sb03md.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# SB03OD benchmark - Discrete-time Lyapunov solver (Cholesky)
bench_sb03od = executable('bench_sb03od',
  'native/bench_sb03od.c',
  include_directories: [inc, bench_inc],
  link_with: slicot_lib,
  dependencies: bench_deps,
  c_args: bench_args,
  install: false,
)

# Meson benchmark registrations
benchmark('bb01ad', bench_bb01ad, timeout: 120)
benchmark('bb02ad', bench_bb02ad, timeout: 120)
benchmark('bb03ad', bench_bb03ad, timeout: 120)
benchmark('bb04ad', bench_bb04ad, timeout: 120)
benchmark('bd01ad', bench_bd01ad, timeout: 120)
benchmark('bd02ad', bench_bd02ad, timeout: 120)

benchmark('sb02md-small', bench_sb02md,
  args: [bench_data_dir / 'BB01103.dat'],
  timeout: 60,
)

benchmark('sb02md-medium', bench_sb02md,
  args: [bench_data_dir / 'BB01104.dat'],
  timeout: 60,
)

benchmark('sb02md-large', bench_sb02md,
  args: [bench_data_dir / 'BB01106.dat'],
  timeout: 60,
)

benchmark('sb02od-small', bench_sb02od,
  args: [bench_data_dir / 'BB02103.dat'],
  timeout: 60,
)

benchmark('sb02od-medium', bench_sb02od,
  args: [bench_data_dir / 'BB02104.dat'],
  timeout: 60,
)

benchmark('sb03md', bench_sb03md, timeout: 120)
benchmark('sb03od', bench_sb03od, timeout: 120)
