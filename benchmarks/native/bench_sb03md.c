/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Benchmark for SB03MD - Continuous-time Lyapunov equation solver
 *
 * Generates test data using BB03AD (CTLEX examples) and solves with SB03MD
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slicot.h"
#include "../common/timing.h"

#define NMAX 200
#define MMAX 200

static void benchmark_sb03md(int scale_param, int ex_num) {
    int n = NMAX, m = MMAX;
    int ldwork_gen = 2 * NMAX;
    int ldwork_solve = n > 1 ? 2 * n * n : 2;

    f64 *e = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *a = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *y = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *b = (f64 *)calloc(MMAX * NMAX, sizeof(f64));
    f64 *x = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *u_gen = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *dwork_gen = (f64 *)calloc(ldwork_gen, sizeof(f64));

    f64 *a_work = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *u_solve = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *c_solve = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *wr = (f64 *)calloc(NMAX, sizeof(f64));
    f64 *wi = (f64 *)calloc(NMAX, sizeof(f64));
    i32 *iwork = (i32 *)calloc(NMAX * NMAX, sizeof(i32));
    f64 *dwork_solve = (f64 *)calloc(ldwork_solve > n * n + 4 * n ? ldwork_solve : n * n + 4 * n, sizeof(f64));

    f64 dpar[2];
    i32 ipar[1] = {scale_param};
    i32 nr[2] = {4, ex_num};
    bool vec[8] = {0};
    char note[80] = {0};
    i32 info;

    if (!e || !a || !y || !b || !x || !u_gen || !dwork_gen ||
        !a_work || !u_solve || !c_solve || !wr || !wi || !iwork || !dwork_solve) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    switch (ex_num) {
        case 1: dpar[0] = 1.5; dpar[1] = 1.5; break;
        case 2: dpar[0] = -0.5; dpar[1] = 1.5; break;
        case 3: dpar[0] = 10.0; dpar[1] = 0.0; break;
        case 4: dpar[0] = 1.5; dpar[1] = 0.0; break;
        default: dpar[0] = 1.5; dpar[1] = 1.5; break;
    }

    /* Generate test data */
    bb03ad("N", nr, dpar, ipar, vec, &n, &m,
           e, NMAX, a, NMAX, y, NMAX, b, MMAX,
           x, NMAX, u_gen, NMAX, note, dwork_gen, ldwork_gen, &info);

    if (info != 0) {
        fprintf(stderr, "BB03AD failed for ex%d with scale=%d, info=%d\n", ex_num, scale_param, info);
        goto cleanup;
    }

    ldwork_solve = 2 * n * n > 2 ? 2 * n * n : 2;
    f64 *dwork_solve_real = (f64 *)realloc(dwork_solve, ldwork_solve * sizeof(f64));
    if (!dwork_solve_real) {
        fprintf(stderr, "Realloc failed\n");
        goto cleanup;
    }
    dwork_solve = dwork_solve_real;

    double times[BENCH_ITERATIONS];
    bench_stats_t stats;
    f64 scale, sep, ferr;

    /* Warmup runs */
    for (int w = 0; w < BENCH_WARMUP; w++) {
        memcpy(a_work, a, n * n * sizeof(f64));
        memcpy(c_solve, y, n * n * sizeof(f64));
        memset(u_solve, 0, n * n * sizeof(f64));

        sb03md("C", "X", "N", "N", n, a_work, n,
               u_solve, n, c_solve, n, &scale,
               &sep, &ferr, wr, wi, iwork,
               dwork_solve, ldwork_solve, &info);
    }

    /* Timed runs */
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        memcpy(a_work, a, n * n * sizeof(f64));
        memcpy(c_solve, y, n * n * sizeof(f64));
        memset(u_solve, 0, n * n * sizeof(f64));

        uint64_t start = bench_time_ns();
        sb03md("C", "X", "N", "N", n, a_work, n,
               u_solve, n, c_solve, n, &scale,
               &sep, &ferr, wr, wi, iwork,
               dwork_solve, ldwork_solve, &info);
        uint64_t end = bench_time_ns();

        times[i] = bench_elapsed_us(start, end);
    }

    bench_compute_stats(times, BENCH_ITERATIONS, &stats);

    printf("sb03md,ctlex4.%d_n%d,n=%d,%.2f,%.2f,%.2f,%.2f,%d\n",
           ex_num, scale_param, n, stats.mean_us, stats.min_us, stats.max_us, stats.stddev_us, info);

cleanup:
    free(e); free(a); free(y); free(b); free(x); free(u_gen); free(dwork_gen);
    free(a_work); free(u_solve); free(c_solve); free(wr); free(wi); free(iwork); free(dwork_solve);
}

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
    printf("routine,dataset,n,mean_us,min_us,max_us,stddev_us,info\n");

    /* Run various sizes from BB03AD group 4 examples */
    for (int scale = 10; scale <= 50; scale += 10) {
        for (int ex = 1; ex <= 4; ex++) {
            benchmark_sb03md(scale, ex);
        }
    }

    return 0;
}
