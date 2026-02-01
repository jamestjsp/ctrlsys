#!/usr/bin/env python3
"""
C11 vs Fortran77 Benchmark Comparison for SLICOT

Runs comprehensive benchmarks comparing the C11 translation against
the original Fortran77 reference implementation.

Usage:
    python scripts/benchmark_c_vs_fortran.py [--output report.md] [--c-build-dir build]
"""

import argparse
import subprocess
import sys
import os
import time
import statistics
import platform
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Configuration
FORTRAN_LIB = "SLICOT-Reference/build_bench/lib/libslicot.dylib"
C_BUILD_DIR = "build"
BENCH_DATA_DIR = "SLICOT-Reference/benchmark_data"
WARMUP_RUNS = 3
TIMED_RUNS = 10

@dataclass
class BenchmarkResult:
    routine: str
    dataset: str
    n: int
    impl: str  # "C11" or "F77"
    mean_us: float
    min_us: float
    max_us: float
    stddev_us: float
    info: int
    raw_times: List[float] = field(default_factory=list)

@dataclass
class ComparisonResult:
    routine: str
    dataset: str
    n: int
    c_mean_us: float
    f_mean_us: float
    speedup: float  # >1 means C is faster
    c_stddev: float
    f_stddev: float
    c_info: int
    f_info: int


def get_system_info() -> Dict[str, str]:
    """Gather system information for the report."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Get CPU info
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info["cpu"] = result.stdout.strip()

            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info["cores"] = result.stdout.strip()

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                info["memory_gb"] = f"{mem_bytes / (1024**3):.1f}"
        except Exception:
            pass

    # Get compiler versions
    try:
        result = subprocess.run(["cc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["c_compiler"] = result.stdout.split('\n')[0]
    except Exception:
        pass

    try:
        result = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            info["fortran_compiler"] = result.stdout.split('\n')[0]
    except Exception:
        pass

    return info


def run_c_benchmark(routine: str, args: List[str] = None) -> List[BenchmarkResult]:
    """Run C benchmark executable and parse results."""
    exe = Path(C_BUILD_DIR) / "benchmarks" / f"bench_{routine}"
    if not exe.exists():
        print(f"Warning: C benchmark {exe} not found", file=sys.stderr)
        return []

    cmd = [str(exe)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout:
        print(f"Warning: {exe.name} failed: {result.stderr}", file=sys.stderr)
        return []

    results = []
    for line in result.stdout.strip().split('\n'):
        if not line or line.startswith('routine,'):
            continue
        parts = line.split(',')
        if len(parts) >= 8:
            try:
                n_str = parts[2]
                n = int(n_str.split('=')[-1]) if '=' in n_str else int(n_str)
                results.append(BenchmarkResult(
                    routine=parts[0],
                    dataset=parts[1],
                    n=n,
                    impl="C11",
                    mean_us=float(parts[3]),
                    min_us=float(parts[4]),
                    max_us=float(parts[5]),
                    stddev_us=float(parts[6]),
                    info=int(parts[7])
                ))
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line} ({e})", file=sys.stderr)

    return results


def create_fortran_benchmark(routine: str, temp_dir: Path) -> Optional[Path]:
    """Create a Fortran benchmark driver for a routine."""

    # Map routines to their benchmark drivers
    drivers = {
        "bb01ad": """
      PROGRAM BENCH_BB01AD
      INTEGER NMAX, MMAX, PMAX
      PARAMETER (NMAX=100, MMAX=100, PMAX=100)
      CHARACTER DEF
      INTEGER NR(2), IPAR(4), N, M, P, INFO, LDWORK, I, W, NR1, NR2
      DOUBLE PRECISION DPAR(7), A(NMAX,NMAX), B(NMAX,MMAX)
      DOUBLE PRECISION C(PMAX,NMAX), G(NMAX,NMAX), Q(NMAX,NMAX)
      DOUBLE PRECISION X(NMAX,NMAX), DWORK(NMAX*NMAX)
      LOGICAL BPAR(6), VEC(9)
      CHARACTER*255 CHPAR
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      DEF = 'D'
      LDWORK = NMAX*NMAX
      DO I = 1, 6
        BPAR(I) = .TRUE.
      END DO
      BPAR(3) = .FALSE.
      BPAR(6) = .FALSE.

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Group 1: Fixed examples
      DO NR2 = 1, 6
        NR(1) = 1
        NR(2) = NR2
        N = NMAX
        M = MMAX
        P = PMAX
        DO W = 1, 3
          CALL BB01AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                G, NMAX, Q, NMAX, X, NMAX, DWORK, LDWORK, INFO)
        END DO
        DO I = 1, 10
          CALL CPU_TIME(T1)
          CALL BB01AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                G, NMAX, Q, NMAX, X, NMAX, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO
        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I1,A,I1,A,I3,4(A,F8.2),A,I2)')
     $    'bb01ad,ex',NR(1),'.',NR(2),',n=',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

C     Group 2: Parameter-dependent
      DO NR2 = 1, 9
        NR(1) = 2
        NR(2) = NR2
        N = NMAX
        M = MMAX
        P = PMAX
        DO W = 1, 3
          CALL BB01AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                G, NMAX, Q, NMAX, X, NMAX, DWORK, LDWORK, INFO)
        END DO
        DO I = 1, 10
          CALL CPU_TIME(T1)
          CALL BB01AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                G, NMAX, Q, NMAX, X, NMAX, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO
        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I1,A,I1,A,I3,4(A,F8.2),A,I2)')
     $    'bb01ad,ex',NR(1),'.',NR(2),',n=',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "sb02md": """
      PROGRAM BENCH_SB02MD
      INTEGER NMAX
      PARAMETER (NMAX=100)
      CHARACTER DICO, HINV, UPLO, SCAL, SORT
      INTEGER N, LDA, LDG, LDQ, LDS, LDU, LDWORK, INFO, I, W
      INTEGER IWORK(2*NMAX), BWORK(2*NMAX)
      DOUBLE PRECISION A(NMAX,NMAX), G(NMAX,NMAX), Q(NMAX,NMAX)
      DOUBLE PRECISION RCOND, WR(2*NMAX), WI(2*NMAX)
      DOUBLE PRECISION S(2*NMAX,2*NMAX), U(2*NMAX,2*NMAX)
      DOUBLE PRECISION DWORK(6*NMAX)
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      DICO = 'C'
      HINV = 'D'
      UPLO = 'U'
      SCAL = 'N'
      SORT = 'S'

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Test with various sizes
      DO N = 10, 50, 10
        LDA = NMAX
        LDG = NMAX
        LDQ = NMAX
        LDS = 2*NMAX
        LDU = 2*NMAX
        LDWORK = 6*N

C       Initialize test matrices
        DO I = 1, N
          DO W = 1, N
            A(I,W) = 0.0D0
            G(I,W) = 0.0D0
            Q(I,W) = 0.0D0
          END DO
          A(I,I) = -1.0D0 - DBLE(I)/DBLE(N)
          G(I,I) = 1.0D0
          Q(I,I) = 1.0D0
        END DO

C       Warmup
        DO W = 1, 3
          CALL SB02MD(DICO, HINV, UPLO, SCAL, SORT, N,
     $                A, LDA, G, LDG, Q, LDQ, RCOND,
     $                WR, WI, S, LDS, U, LDU,
     $                IWORK, DWORK, LDWORK, BWORK, INFO)
        END DO

C       Timed runs
        DO I = 1, 10
          DO W = 1, N
            A(W,W) = -1.0D0 - DBLE(W)/DBLE(N)
          END DO
          CALL CPU_TIME(T1)
          CALL SB02MD(DICO, HINV, UPLO, SCAL, SORT, N,
     $                A, LDA, G, LDG, Q, LDQ, RCOND,
     $                WR, WI, S, LDS, U, LDU,
     $                IWORK, DWORK, LDWORK, BWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO

        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,A,I3,4(A,F10.2),A,I2)')
     $    'sb02md,synthetic,n=',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "bb02ad": """
      PROGRAM BENCH_BB02AD
      INTEGER NMAX, MMAX, PMAX
      PARAMETER (NMAX=100, MMAX=100, PMAX=100)
      CHARACTER DEF
      INTEGER NR(2), IPAR(3), N, M, P, INFO, LDWORK, I, W, NR1, NR2
      DOUBLE PRECISION DPAR(4), A(NMAX,NMAX), B(NMAX,MMAX)
      DOUBLE PRECISION C(PMAX,NMAX), Q(NMAX,NMAX), R(MMAX,MMAX)
      DOUBLE PRECISION S(NMAX,MMAX), X(NMAX,NMAX), DWORK(NMAX*NMAX)
      LOGICAL BPAR(7), VEC(10)
      CHARACTER*255 CHPAR
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      DEF = 'D'
      LDWORK = NMAX*NMAX
      DO I = 1, 7
        BPAR(I) = .TRUE.
      END DO
      BPAR(3) = .FALSE.
      BPAR(6) = .FALSE.
      BPAR(7) = .FALSE.

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Group 1: Fixed examples (1-13)
      DO NR2 = 1, 13
        NR(1) = 1
        NR(2) = NR2
        N = NMAX
        M = MMAX
        P = PMAX
        DO W = 1, 3
          CALL BB02AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                Q, NMAX, R, MMAX, S, NMAX,
     $                X, NMAX, DWORK, LDWORK, INFO)
        END DO
        DO I = 1, 10
          CALL CPU_TIME(T1)
          CALL BB02AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                Q, NMAX, R, MMAX, S, NMAX,
     $                X, NMAX, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO
        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I1,A,I2,A,I3,4(A,F8.2),A,I2)')
     $    'bb02ad,ex',NR(1),'.',NR(2),',n=',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

C     Group 2: Parameter-dependent (1-5)
      DO NR2 = 1, 5
        NR(1) = 2
        NR(2) = NR2
        N = NMAX
        M = MMAX
        P = PMAX
        DO W = 1, 3
          CALL BB02AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                Q, NMAX, R, MMAX, S, NMAX,
     $                X, NMAX, DWORK, LDWORK, INFO)
        END DO
        DO I = 1, 10
          CALL CPU_TIME(T1)
          CALL BB02AD(DEF, NR, DPAR, IPAR, BPAR, CHPAR, VEC,
     $                N, M, P, A, NMAX, B, NMAX, C, PMAX,
     $                Q, NMAX, R, MMAX, S, NMAX,
     $                X, NMAX, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO
        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I1,A,I1,A,I3,4(A,F8.2),A,I2)')
     $    'bb02ad,ex',NR(1),'.',NR(2),',n=',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "sb03md": """
      PROGRAM BENCH_SB03MD
      INTEGER NMAX
      PARAMETER (NMAX=100)
      CHARACTER DICO, JOB, FACT, TRANA
      INTEGER N, LDA, LDU, LDC, LDWORK, INFO, I, W
      INTEGER IWORK(NMAX*NMAX)
      DOUBLE PRECISION A(NMAX,NMAX), U(NMAX,NMAX), C(NMAX,NMAX)
      DOUBLE PRECISION SCALE, SEP, FERR, WR(NMAX), WI(NMAX)
      DOUBLE PRECISION DWORK(2*NMAX*NMAX)
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      DICO = 'C'
      JOB = 'X'
      FACT = 'N'
      TRANA = 'N'

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Test with various sizes
      DO N = 10, 50, 10
        LDA = NMAX
        LDU = NMAX
        LDC = NMAX
        LDWORK = 2*N*N

C       Initialize test matrices (stable A, symmetric C)
        DO I = 1, N
          DO W = 1, N
            A(I,W) = 0.0D0
            C(I,W) = 0.0D0
            U(I,W) = 0.0D0
          END DO
          A(I,I) = -1.0D0 - DBLE(I)/DBLE(N)
          C(I,I) = 1.0D0
        END DO

C       Warmup
        DO W = 1, 3
          DO I = 1, N
            A(I,I) = -1.0D0 - DBLE(I)/DBLE(N)
            C(I,I) = 1.0D0
          END DO
          CALL SB03MD(DICO, JOB, FACT, TRANA, N,
     $                A, LDA, U, LDU, C, LDC, SCALE,
     $                SEP, FERR, WR, WI, IWORK,
     $                DWORK, LDWORK, INFO)
        END DO

C       Timed runs
        DO I = 1, 10
          DO W = 1, N
            A(W,W) = -1.0D0 - DBLE(W)/DBLE(N)
            C(W,W) = 1.0D0
          END DO
          CALL CPU_TIME(T1)
          CALL SB03MD(DICO, JOB, FACT, TRANA, N,
     $                A, LDA, U, LDU, C, LDC, SCALE,
     $                SEP, FERR, WR, WI, IWORK,
     $                DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO

        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I2,A,I3,4(A,F10.2),A,I2)')
     $    'sb03md,synthetic_n',N,',',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "bb03ad": """
      PROGRAM BENCH_BB03AD
      INTEGER NMAX, MMAX
      PARAMETER (NMAX=150, MMAX=150)
      CHARACTER DEF
      INTEGER NR(2), IPAR(1), N, M, INFO, LDWORK, I, W, SCALE, EX
      DOUBLE PRECISION DPAR(2), E(NMAX,NMAX), A(NMAX,NMAX)
      DOUBLE PRECISION Y(NMAX,NMAX), B(MMAX,NMAX)
      DOUBLE PRECISION X(NMAX,NMAX), U(NMAX,NMAX), DWORK(2*NMAX)
      LOGICAL VEC(8)
      CHARACTER*80 NOTE
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      LDWORK = 2*NMAX
      DO I = 1, 8
        VEC(I) = .FALSE.
      END DO

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Group 4: Scalable with different sub-examples
      DO SCALE = 10, 50, 10
        DO EX = 1, 4
          DEF = 'N'
          NR(1) = 4
          NR(2) = EX
          IPAR(1) = SCALE
          N = NMAX
          M = MMAX
          IF (EX .EQ. 1) THEN
            DPAR(1) = 1.5D0
            DPAR(2) = 1.5D0
          ELSE IF (EX .EQ. 2) THEN
            DPAR(1) = -0.5D0
            DPAR(2) = 1.5D0
          ELSE IF (EX .EQ. 3) THEN
            DPAR(1) = 10.0D0
            DPAR(2) = 0.0D0
          ELSE
            DPAR(1) = 1.5D0
            DPAR(2) = 0.0D0
          END IF
          DO W = 1, 3
            CALL BB03AD(DEF, NR, DPAR, IPAR, VEC, N, M,
     $                  E, NMAX, A, NMAX, Y, NMAX, B, MMAX,
     $                  X, NMAX, U, NMAX, NOTE, DWORK, LDWORK, INFO)
          END DO
          DO I = 1, 10
            N = NMAX
            M = MMAX
            CALL CPU_TIME(T1)
            CALL BB03AD(DEF, NR, DPAR, IPAR, VEC, N, M,
     $                  E, NMAX, A, NMAX, Y, NMAX, B, MMAX,
     $                  X, NMAX, U, NMAX, NOTE, DWORK, LDWORK, INFO)
            CALL CPU_TIME(T2)
            TIMES(I) = (T2-T1)*1.0D6
          END DO
          CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
          WRITE(*,'(A,I1,A,I3,4(A,F8.2),A,I2)')
     $      'bb03ad,ex4.',EX,',',N,
     $      ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
        END DO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "bb04ad": """
      PROGRAM BENCH_BB04AD
      INTEGER NMAX, MMAX
      PARAMETER (NMAX=150, MMAX=150)
      CHARACTER DEF
      INTEGER NR(2), IPAR(1), N, M, INFO, LDWORK, I, W, SCALE, EX
      DOUBLE PRECISION DPAR(2), E(NMAX,NMAX), A(NMAX,NMAX)
      DOUBLE PRECISION Y(NMAX,NMAX), B(MMAX,NMAX)
      DOUBLE PRECISION X(NMAX,NMAX), U(NMAX,NMAX), DWORK(2*NMAX)
      LOGICAL VEC(8)
      CHARACTER*80 NOTE
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      LDWORK = 2*NMAX
      DO I = 1, 8
        VEC(I) = .FALSE.
      END DO

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Group 4: Scalable
      DO SCALE = 10, 50, 10
        DO EX = 1, 4
          DEF = 'N'
          NR(1) = 4
          NR(2) = EX
          IPAR(1) = SCALE
          N = NMAX
          M = MMAX
          IF (EX .EQ. 1) THEN
            DPAR(1) = 1.5D0
            DPAR(2) = 1.5D0
          ELSE IF (EX .EQ. 2) THEN
            DPAR(1) = 0.5D0
            DPAR(2) = 1.5D0
          ELSE IF (EX .EQ. 3) THEN
            DPAR(1) = 10.0D0
            DPAR(2) = 0.0D0
          ELSE
            DPAR(1) = 1.5D0
            DPAR(2) = 0.0D0
          END IF
          DO W = 1, 3
            CALL BB04AD(DEF, NR, DPAR, IPAR, VEC, N, M,
     $                  E, NMAX, A, NMAX, Y, NMAX, B, MMAX,
     $                  X, NMAX, U, NMAX, NOTE, DWORK, LDWORK, INFO)
          END DO
          DO I = 1, 10
            N = NMAX
            M = MMAX
            CALL CPU_TIME(T1)
            CALL BB04AD(DEF, NR, DPAR, IPAR, VEC, N, M,
     $                  E, NMAX, A, NMAX, Y, NMAX, B, MMAX,
     $                  X, NMAX, U, NMAX, NOTE, DWORK, LDWORK, INFO)
            CALL CPU_TIME(T2)
            TIMES(I) = (T2-T1)*1.0D6
          END DO
          CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
          WRITE(*,'(A,I1,A,I3,4(A,F8.2),A,I2)')
     $      'bb04ad,ex4.',EX,',',N,
     $      ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
        END DO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "bd01ad": """
      PROGRAM BENCH_BD01AD
      INTEGER NMAX, MMAX, PMAX
      PARAMETER (NMAX=100, MMAX=100, PMAX=100)
      CHARACTER DEF
      INTEGER NR(2), IPAR(7), N, M, P, INFO, LDWORK, I, W, EX
      DOUBLE PRECISION DPAR(7), E(NMAX,NMAX), A(NMAX,NMAX)
      DOUBLE PRECISION B(NMAX,MMAX), C(PMAX,NMAX), D(PMAX,MMAX)
      DOUBLE PRECISION DWORK(NMAX*NMAX)
      LOGICAL VEC(8)
      CHARACTER*256 CHPAR
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      DEF = 'D'
      LDWORK = NMAX*NMAX
      DO I = 1, 8
        VEC(I) = .FALSE.
      END DO
      DO I = 1, 7
        IPAR(I) = 0
        DPAR(I) = 0.0D0
      END DO

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Group 1: Fixed-size examples
      DO EX = 1, 10
        NR(1) = 1
        NR(2) = EX
        N = NMAX
        M = MMAX
        P = PMAX
        DO W = 1, 3
          CALL BD01AD(DEF, NR, DPAR, IPAR, VEC, N, M, P,
     $                E, NMAX, A, NMAX, B, NMAX, C, PMAX, D, PMAX,
     $                CHPAR, DWORK, LDWORK, INFO)
        END DO
        DO I = 1, 10
          N = NMAX
          M = MMAX
          P = PMAX
          CALL CPU_TIME(T1)
          CALL BD01AD(DEF, NR, DPAR, IPAR, VEC, N, M, P,
     $                E, NMAX, A, NMAX, B, NMAX, C, PMAX, D, PMAX,
     $                CHPAR, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO
        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I2,A,I3,4(A,F8.2),A,I2)')
     $    'bd01ad,ex1.',EX,',',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

C     Group 2: Parameter-dependent
      DO EX = 1, 6
        NR(1) = 2
        NR(2) = EX
        N = NMAX
        M = MMAX
        P = PMAX
        DO W = 1, 3
          CALL BD01AD(DEF, NR, DPAR, IPAR, VEC, N, M, P,
     $                E, NMAX, A, NMAX, B, NMAX, C, PMAX, D, PMAX,
     $                CHPAR, DWORK, LDWORK, INFO)
        END DO
        DO I = 1, 10
          N = NMAX
          M = MMAX
          P = PMAX
          CALL CPU_TIME(T1)
          CALL BD01AD(DEF, NR, DPAR, IPAR, VEC, N, M, P,
     $                E, NMAX, A, NMAX, B, NMAX, C, PMAX, D, PMAX,
     $                CHPAR, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO
        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I1,A,I3,4(A,F8.2),A,I2)')
     $    'bd01ad,ex2.',EX,',',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "bd02ad": """
      PROGRAM BENCH_BD02AD
      INTEGER NMAX, MMAX, PMAX
      PARAMETER (NMAX=100, MMAX=100, PMAX=100)
      CHARACTER DEF
      INTEGER NR(2), IPAR(7), N, M, P, INFO, LDWORK, I, W, EX
      DOUBLE PRECISION DPAR(7), E(NMAX,NMAX), A(NMAX,NMAX)
      DOUBLE PRECISION B(NMAX,MMAX), C(PMAX,NMAX), D(PMAX,MMAX)
      DOUBLE PRECISION DWORK(NMAX*NMAX)
      LOGICAL VEC(8)
      CHARACTER*80 NOTE
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      DEF = 'D'
      LDWORK = NMAX*NMAX
      DO I = 1, 8
        VEC(I) = .FALSE.
      END DO
      DO I = 1, 7
        IPAR(I) = 0
        DPAR(I) = 0.0D0
      END DO

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Group 1: Fixed-size examples
      DO EX = 1, 12
        NR(1) = 1
        NR(2) = EX
        N = NMAX
        M = MMAX
        P = PMAX
        DO W = 1, 3
          CALL BD02AD(DEF, NR, DPAR, IPAR, VEC, N, M, P,
     $                E, NMAX, A, NMAX, B, NMAX, C, PMAX, D, PMAX,
     $                NOTE, DWORK, LDWORK, INFO)
        END DO
        DO I = 1, 10
          N = NMAX
          M = MMAX
          P = PMAX
          CALL CPU_TIME(T1)
          CALL BD02AD(DEF, NR, DPAR, IPAR, VEC, N, M, P,
     $                E, NMAX, A, NMAX, B, NMAX, C, PMAX, D, PMAX,
     $                NOTE, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO
        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I2,A,I3,4(A,F8.2),A,I2)')
     $    'bd02ad,ex1.',EX,',',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
""",
        "sb03od": """
      PROGRAM BENCH_SB03OD
      INTEGER NMAX
      PARAMETER (NMAX=100)
      CHARACTER DICO, FACT, TRANS
      INTEGER N, LDA, LDQ, LDB, LDWORK, INFO, I, W, MSOL
      DOUBLE PRECISION A(NMAX,NMAX), Q(NMAX,NMAX), B(NMAX,NMAX)
      DOUBLE PRECISION SCALE, WR(NMAX), WI(NMAX)
      DOUBLE PRECISION DWORK(4*NMAX)
      DOUBLE PRECISION T1, T2, TIMES(10), MEAN, TMIN, TMAX, STD

      DICO = 'D'
      FACT = 'N'
      TRANS = 'N'

      WRITE(*,'(A)') 'routine,example,size,mean_us,min_us,max_us,'//
     $               'stddev_us,info'

C     Test with various sizes
      DO N = 10, 50, 10
        LDA = NMAX
        LDQ = NMAX
        LDB = NMAX
        LDWORK = 4*N
        MSOL = N

C       Initialize: stable A (eigenvalues inside unit circle), B = I
        DO I = 1, N
          DO W = 1, N
            A(I,W) = 0.0D0
            Q(I,W) = 0.0D0
            B(I,W) = 0.0D0
          END DO
          A(I,I) = 0.5D0 - 0.3D0*DBLE(I)/DBLE(N)
          B(I,I) = 1.0D0
        END DO

C       Warmup
        DO W = 1, 3
          DO I = 1, N
            A(I,I) = 0.5D0 - 0.3D0*DBLE(I)/DBLE(N)
            B(I,I) = 1.0D0
          END DO
          CALL SB03OD(DICO, FACT, TRANS, N, MSOL,
     $                A, LDA, Q, LDQ, B, LDB,
     $                SCALE, WR, WI, DWORK, LDWORK, INFO)
        END DO

C       Timed runs
        DO I = 1, 10
          DO W = 1, N
            A(W,W) = 0.5D0 - 0.3D0*DBLE(W)/DBLE(N)
            B(W,W) = 1.0D0
            Q(W,W) = 0.0D0
          END DO
          CALL CPU_TIME(T1)
          CALL SB03OD(DICO, FACT, TRANS, N, MSOL,
     $                A, LDA, Q, LDQ, B, LDB,
     $                SCALE, WR, WI, DWORK, LDWORK, INFO)
          CALL CPU_TIME(T2)
          TIMES(I) = (T2-T1)*1.0D6
        END DO

        CALL STATS(TIMES, 10, MEAN, TMIN, TMAX, STD)
        WRITE(*,'(A,I2,A,I3,4(A,F10.2),A,I2)')
     $    'sb03od,synthetic_n',N,',',N,
     $    ',',MEAN,',',TMIN,',',TMAX,',',STD,',',INFO
      END DO

      END

      SUBROUTINE STATS(X, N, MEAN, XMIN, XMAX, STD)
      INTEGER N, I
      DOUBLE PRECISION X(N), MEAN, XMIN, XMAX, STD, SUM, SUMSQ
      SUM = 0.0D0
      XMIN = X(1)
      XMAX = X(1)
      DO I = 1, N
        SUM = SUM + X(I)
        IF (X(I) .LT. XMIN) XMIN = X(I)
        IF (X(I) .GT. XMAX) XMAX = X(I)
      END DO
      MEAN = SUM / N
      SUMSQ = 0.0D0
      DO I = 1, N
        SUMSQ = SUMSQ + (X(I) - MEAN)**2
      END DO
      STD = SQRT(SUMSQ / N)
      END
"""
    }

    if routine not in drivers:
        return None

    src_file = temp_dir / f"bench_{routine}.f"
    src_file.write_text(drivers[routine])
    return src_file


def compile_fortran_benchmark(src_file: Path, output: Path) -> bool:
    """Compile a Fortran benchmark driver."""
    fortran_lib = Path(FORTRAN_LIB)
    if not fortran_lib.exists():
        print(f"Error: Fortran library not found: {fortran_lib}", file=sys.stderr)
        return False

    cmd = [
        "gfortran", "-O3", "-o", str(output), str(src_file),
        "-L", str(fortran_lib.parent), "-lslicot",
        "-framework", "Accelerate"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}", file=sys.stderr)
        return False
    return True


def run_fortran_benchmark(exe: Path) -> List[BenchmarkResult]:
    """Run Fortran benchmark executable and parse results."""
    if not exe.exists():
        print(f"Warning: Fortran benchmark {exe} not found", file=sys.stderr)
        return []

    fortran_lib = Path(FORTRAN_LIB)
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(fortran_lib.parent)

    result = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
    if result.returncode != 0 and not result.stdout:
        print(f"Warning: {exe.name} failed: {result.stderr}", file=sys.stderr)
        return []

    results = []
    for line in result.stdout.strip().split('\n'):
        if not line or line.startswith('routine,'):
            continue
        parts = line.split(',')
        if len(parts) >= 8:
            try:
                n_str = parts[2]
                n = int(n_str.split('=')[-1]) if '=' in n_str else int(n_str)
                results.append(BenchmarkResult(
                    routine=parts[0],
                    dataset=parts[1],
                    n=n,
                    impl="F77",
                    mean_us=float(parts[3]),
                    min_us=float(parts[4]),
                    max_us=float(parts[5]),
                    stddev_us=float(parts[6]),
                    info=int(parts[7])
                ))
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line} ({e})", file=sys.stderr)

    return results


def compare_results(c_results: List[BenchmarkResult],
                   f_results: List[BenchmarkResult]) -> List[ComparisonResult]:
    """Compare C and Fortran benchmark results."""
    comparisons = []

    # Index Fortran results by (routine, dataset, n)
    f_index = {}
    for r in f_results:
        key = (r.routine, r.dataset, r.n)
        f_index[key] = r

    for c in c_results:
        key = (c.routine, c.dataset, c.n)
        if key in f_index:
            f = f_index[key]
            speedup = f.mean_us / c.mean_us if c.mean_us > 0 else float('inf')
            comparisons.append(ComparisonResult(
                routine=c.routine,
                dataset=c.dataset,
                n=c.n,
                c_mean_us=c.mean_us,
                f_mean_us=f.mean_us,
                speedup=speedup,
                c_stddev=c.stddev_us,
                f_stddev=f.stddev_us,
                c_info=c.info,
                f_info=f.info
            ))

    return comparisons


def generate_markdown_report(
    c_results: List[BenchmarkResult],
    f_results: List[BenchmarkResult],
    comparisons: List[ComparisonResult],
    system_info: Dict[str, str]
) -> str:
    """Generate detailed markdown benchmark report."""

    lines = []

    # Header
    lines.append("# SLICOT C11 vs Fortran77 Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    if comparisons:
        avg_speedup = statistics.mean([c.speedup for c in comparisons])
        median_speedup = statistics.median([c.speedup for c in comparisons])
        max_speedup = max(c.speedup for c in comparisons)
        min_speedup = min(c.speedup for c in comparisons)

        faster_c = sum(1 for c in comparisons if c.speedup > 1.0)
        faster_f = sum(1 for c in comparisons if c.speedup < 1.0)
        equal = sum(1 for c in comparisons if 0.95 <= c.speedup <= 1.05)

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Comparisons | {len(comparisons)} |")
        lines.append(f"| C11 Faster | {faster_c} ({100*faster_c/len(comparisons):.1f}%) |")
        lines.append(f"| F77 Faster | {faster_f} ({100*faster_f/len(comparisons):.1f}%) |")
        lines.append(f"| Within 5% | {equal} ({100*equal/len(comparisons):.1f}%) |")
        lines.append(f"| Mean Speedup | {avg_speedup:.3f}x |")
        lines.append(f"| Median Speedup | {median_speedup:.3f}x |")
        lines.append(f"| Max Speedup (C11) | {max_speedup:.3f}x |")
        lines.append(f"| Min Speedup (C11) | {min_speedup:.3f}x |")
        lines.append("")

        if avg_speedup > 1.0:
            lines.append(f"> **Result:** C11 implementation is **{avg_speedup:.1f}x faster** on average")
        else:
            lines.append(f"> **Result:** F77 implementation is **{1/avg_speedup:.1f}x faster** on average")
    else:
        lines.append("*No comparison data available*")
    lines.append("")

    # System Information
    lines.append("## System Configuration")
    lines.append("")
    lines.append("### Hardware")
    lines.append("")
    lines.append("| Component | Specification |")
    lines.append("|-----------|---------------|")
    lines.append(f"| CPU | {system_info.get('cpu', 'Unknown')} |")
    lines.append(f"| Cores | {system_info.get('cores', 'Unknown')} |")
    lines.append(f"| Memory | {system_info.get('memory_gb', 'Unknown')} GB |")
    lines.append(f"| Architecture | {system_info.get('architecture', 'Unknown')} |")
    lines.append("")

    lines.append("### Software")
    lines.append("")
    lines.append("| Component | Version |")
    lines.append("|-----------|---------|")
    lines.append(f"| OS | {system_info.get('platform', 'Unknown')} {system_info.get('platform_release', '')} |")
    lines.append(f"| C Compiler | {system_info.get('c_compiler', 'Unknown')} |")
    lines.append(f"| Fortran Compiler | {system_info.get('fortran_compiler', 'Unknown')} |")
    lines.append(f"| BLAS/LAPACK | Apple Accelerate Framework |")
    lines.append("")

    lines.append("### Benchmark Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Warmup Runs | {WARMUP_RUNS} |")
    lines.append(f"| Timed Runs | {TIMED_RUNS} |")
    lines.append(f"| Optimization | `-O3` (both compilers) |")
    lines.append("")

    # Detailed Comparison Tables
    lines.append("## Detailed Results by Routine")
    lines.append("")

    # Group by routine
    routines = sorted(set(c.routine for c in comparisons))

    for routine in routines:
        routine_comps = [c for c in comparisons if c.routine == routine]
        if not routine_comps:
            continue

        lines.append(f"### {routine.upper()}")
        lines.append("")

        # Routine description
        routine_desc = {
            "bb01ad": "CAREX - Continuous-time Algebraic Riccati Equation Examples",
            "bb02ad": "DAREX - Discrete-time Algebraic Riccati Equation Examples",
            "bb03ad": "CTLEX - Continuous-time Lyapunov Equation Examples",
            "bb04ad": "DTLEX - Discrete-time Lyapunov Equation Examples",
            "bd01ad": "CTDSX - Continuous-time Descriptor System Examples",
            "bd02ad": "DTDSX - Discrete-time Descriptor System Examples",
            "sb02md": "Continuous-time Algebraic Riccati Equation Solver",
            "sb02od": "Discrete-time Algebraic Riccati Equation Solver",
            "sb03md": "Continuous-time Lyapunov Equation Solver",
            "sb03od": "Discrete-time Lyapunov Equation Solver",
        }
        lines.append(f"*{routine_desc.get(routine, 'SLICOT Routine')}*")
        lines.append("")

        # Summary for this routine
        avg_sp = statistics.mean([c.speedup for c in routine_comps])
        lines.append(f"**Average Speedup:** {avg_sp:.3f}x {'(C11 faster)' if avg_sp > 1 else '(F77 faster)'}")
        lines.append("")

        # Detailed table
        lines.append("| Dataset | N | C11 (μs) | F77 (μs) | Speedup | C11 σ | F77 σ | Status |")
        lines.append("|---------|---|----------|----------|---------|-------|-------|--------|")

        for c in sorted(routine_comps, key=lambda x: (x.dataset, x.n)):
            speedup_str = f"{c.speedup:.2f}x"
            if c.speedup > 1.1:
                speedup_str = f"**{speedup_str}**"
            elif c.speedup < 0.9:
                speedup_str = f"*{speedup_str}*"

            status = "✓" if c.c_info == 0 and c.f_info == 0 else f"C:{c.c_info}/F:{c.f_info}"

            lines.append(f"| {c.dataset} | {c.n} | {c.c_mean_us:.2f} | {c.f_mean_us:.2f} | "
                        f"{speedup_str} | {c.c_stddev:.2f} | {c.f_stddev:.2f} | {status} |")

        lines.append("")

    # Raw Results
    lines.append("## Raw Benchmark Data")
    lines.append("")

    lines.append("### C11 Implementation Results")
    lines.append("")
    lines.append("```csv")
    lines.append("routine,dataset,n,mean_us,min_us,max_us,stddev_us,info")
    for r in c_results:
        lines.append(f"{r.routine},{r.dataset},{r.n},{r.mean_us:.2f},{r.min_us:.2f},"
                    f"{r.max_us:.2f},{r.stddev_us:.2f},{r.info}")
    lines.append("```")
    lines.append("")

    lines.append("### Fortran77 Implementation Results")
    lines.append("")
    lines.append("```csv")
    lines.append("routine,dataset,n,mean_us,min_us,max_us,stddev_us,info")
    for r in f_results:
        lines.append(f"{r.routine},{r.dataset},{r.n},{r.mean_us:.2f},{r.min_us:.2f},"
                    f"{r.max_us:.2f},{r.stddev_us:.2f},{r.info}")
    lines.append("```")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Timing Approach")
    lines.append("")
    lines.append("- **C11:** High-resolution platform timers (`mach_absolute_time` on macOS)")
    lines.append("- **Fortran77:** `CPU_TIME` intrinsic subroutine")
    lines.append("- Both implementations linked against identical BLAS/LAPACK (Apple Accelerate)")
    lines.append("- Identical test matrices and problem sizes used")
    lines.append("")

    lines.append("### Statistical Analysis")
    lines.append("")
    lines.append("- Mean, min, max, and standard deviation computed over timed runs")
    lines.append("- Speedup = F77_mean / C11_mean (>1 means C11 is faster)")
    lines.append("- Warmup runs excluded from timing statistics")
    lines.append("")

    lines.append("### Caveats")
    lines.append("")
    lines.append("1. CPU frequency scaling may affect results")
    lines.append("2. Memory caching effects present in both implementations")
    lines.append("3. Small timing variations expected between runs")
    lines.append("4. Some routines may show different behavior due to algorithm variations")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `scripts/benchmark_c_vs_fortran.py`*")

    return "\n".join(lines)


def main():
    global C_BUILD_DIR
    parser = argparse.ArgumentParser(description="SLICOT C11 vs Fortran77 Benchmark")
    parser.add_argument("--output", "-o", default="benchmark_report.md",
                       help="Output markdown file")
    parser.add_argument("--routines", "-r", nargs="+",
                       default=["bb01ad", "bb02ad", "bb03ad", "bb04ad",
                                "bd01ad", "bd02ad", "sb03md", "sb03od"],
                       help="Routines to benchmark")
    parser.add_argument("--c-build-dir", default="build",
                       help="C build directory (default: build)")
    args = parser.parse_args()
    C_BUILD_DIR = args.c_build_dir

    print("=" * 60)
    print("SLICOT C11 vs Fortran77 Benchmark Comparison")
    print("=" * 60)
    print()

    # Check prerequisites
    if not Path(FORTRAN_LIB).exists():
        print(f"Error: Fortran library not found at {FORTRAN_LIB}")
        print("Build it with: cd SLICOT-Reference && cmake -B build_bench && cmake --build build_bench")
        return 1

    if not Path(C_BUILD_DIR).exists():
        print(f"Error: C build directory not found at {C_BUILD_DIR}")
        print("Build it with: meson setup build && meson compile -C build")
        return 1

    # Gather system info
    print("Gathering system information...")
    system_info = get_system_info()

    # Create temp directory for Fortran sources
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        all_c_results = []
        all_f_results = []

        for routine in args.routines:
            print(f"\nBenchmarking {routine.upper()}...")

            # Run C benchmark
            print(f"  Running C11 benchmark...")
            c_results = run_c_benchmark(routine)
            all_c_results.extend(c_results)
            print(f"    Got {len(c_results)} results")

            # Create and compile Fortran benchmark
            print(f"  Creating Fortran benchmark...")
            src_file = create_fortran_benchmark(routine, temp_path)
            if src_file:
                exe_file = temp_path / f"bench_{routine}_f77"
                print(f"  Compiling Fortran benchmark...")
                if compile_fortran_benchmark(src_file, exe_file):
                    print(f"  Running Fortran benchmark...")
                    f_results = run_fortran_benchmark(exe_file)
                    all_f_results.extend(f_results)
                    print(f"    Got {len(f_results)} results")
                else:
                    print(f"    Compilation failed")
            else:
                print(f"    No Fortran driver available for {routine}")

        # Compare results
        print("\nComparing results...")
        comparisons = compare_results(all_c_results, all_f_results)

        # Generate report
        print(f"\nGenerating report: {args.output}")
        report = generate_markdown_report(all_c_results, all_f_results,
                                         comparisons, system_info)

        Path(args.output).write_text(report)
        print(f"\nReport written to {args.output}")

        # Print summary
        if comparisons:
            avg_speedup = statistics.mean([c.speedup for c in comparisons])
            print(f"\nSummary: Average speedup = {avg_speedup:.2f}x")
            if avg_speedup > 1:
                print(f"         C11 is {avg_speedup:.1f}x faster on average")
            else:
                print(f"         F77 is {1/avg_speedup:.1f}x faster on average")

    return 0


if __name__ == "__main__":
    sys.exit(main())
