/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Benchmark for SB03OD - Discrete-time Lyapunov equation solver (Cholesky factor)
 *
 * Generates test data using BB04AD (DTLEX examples) and solves with SB03OD
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slicot.h"
#include "../common/timing.h"

#define NMAX 200
#define MMAX 200

static void benchmark_sb03od(int scale_param, int ex_num) {
    int n = NMAX, m = MMAX;
    int ldwork_gen = 2 * NMAX;

    f64 *e = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *a = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *y = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *b_gen = (f64 *)calloc(MMAX * NMAX, sizeof(f64));
    f64 *x = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *u_gen = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *dwork_gen = (f64 *)calloc(ldwork_gen, sizeof(f64));

    f64 *a_work = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *q_solve = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *b_solve = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *wr = (f64 *)calloc(NMAX, sizeof(f64));
    f64 *wi = (f64 *)calloc(NMAX, sizeof(f64));
    f64 *dwork_solve = NULL;

    f64 dpar[2];
    i32 ipar[1] = {scale_param};
    i32 nr[2] = {4, ex_num};
    bool vec[8] = {0};
    char note[80] = {0};
    i32 info;

    if (!e || !a || !y || !b_gen || !x || !u_gen || !dwork_gen ||
        !a_work || !q_solve || !b_solve || !wr || !wi) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    switch (ex_num) {
        case 1: dpar[0] = 1.5; dpar[1] = 1.5; break;
        case 2: dpar[0] = 0.5; dpar[1] = 1.5; break;
        case 3: dpar[0] = 10.0; dpar[1] = 0.0; break;
        case 4: dpar[0] = 1.5; dpar[1] = 0.0; break;
        default: dpar[0] = 1.5; dpar[1] = 1.5; break;
    }

    /* Generate test data */
    bb04ad("N", nr, dpar, ipar, vec, &n, &m,
           e, NMAX, a, NMAX, y, NMAX, b_gen, MMAX,
           x, NMAX, u_gen, NMAX, note, dwork_gen, ldwork_gen, &info);

    if (info != 0) {
        fprintf(stderr, "BB04AD failed for ex%d with scale=%d, info=%d\n", ex_num, scale_param, info);
        goto cleanup;
    }

    int ldwork_solve = 4 * n > 1 ? 4 * n : 1;
    dwork_solve = (f64 *)calloc(ldwork_solve, sizeof(f64));
    if (!dwork_solve) {
        fprintf(stderr, "Workspace allocation failed\n");
        goto cleanup;
    }

    /* Form B from Y for SB03OD: Y = -B'*B, need B upper triangular */
    int m_solve = n;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            b_solve[i + j * n] = (i == j) ? 1.0 : 0.0;
        }
    }

    double times[BENCH_ITERATIONS];
    bench_stats_t stats;
    f64 scale;

    /* Warmup runs */
    for (int w = 0; w < BENCH_WARMUP; w++) {
        memcpy(a_work, a, n * n * sizeof(f64));
        memset(q_solve, 0, n * n * sizeof(f64));
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                b_solve[i + j * n] = (i == j) ? 1.0 : 0.0;
            }
        }

        sb03od("D", "N", "N", n, m_solve,
               a_work, n, q_solve, n, b_solve, n,
               &scale, wr, wi, dwork_solve, ldwork_solve, &info);
    }

    /* Timed runs */
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        memcpy(a_work, a, n * n * sizeof(f64));
        memset(q_solve, 0, n * n * sizeof(f64));
        for (int j = 0; j < n; j++) {
            for (int k = j; k < n; k++) {
                b_solve[j + k * n] = (j == k) ? 1.0 : 0.0;
            }
        }

        uint64_t start = bench_time_ns();
        sb03od("D", "N", "N", n, m_solve,
               a_work, n, q_solve, n, b_solve, n,
               &scale, wr, wi, dwork_solve, ldwork_solve, &info);
        uint64_t end = bench_time_ns();

        times[i] = bench_elapsed_us(start, end);
    }

    bench_compute_stats(times, BENCH_ITERATIONS, &stats);

    printf("sb03od,dtlex4.%d_n%d,n=%d,%.2f,%.2f,%.2f,%.2f,%d\n",
           ex_num, scale_param, n, stats.mean_us, stats.min_us, stats.max_us, stats.stddev_us, info);

cleanup:
    free(e); free(a); free(y); free(b_gen); free(x); free(u_gen); free(dwork_gen);
    free(a_work); free(q_solve); free(b_solve); free(wr); free(wi); free(dwork_solve);
}

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
    printf("routine,dataset,n,mean_us,min_us,max_us,stddev_us,info\n");

    /* Run various sizes from BB04AD group 4 examples */
    for (int scale = 10; scale <= 50; scale += 10) {
        for (int ex = 1; ex <= 4; ex++) {
            benchmark_sb03od(scale, ex);
        }
    }

    return 0;
}
