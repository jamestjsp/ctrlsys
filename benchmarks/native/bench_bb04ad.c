/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Benchmark for BB04AD - DTLEX benchmark example generator (discrete Lyapunov)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slicot.h"
#include "../common/timing.h"

#define NMAX 150
#define MMAX 150

static void benchmark_bb04ad(int nr2, int scale_param) {
    int n = NMAX, m = MMAX;
    int ldwork = 2 * NMAX;

    f64 *e = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *a = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *y = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *b = (f64 *)calloc(MMAX * NMAX, sizeof(f64));
    f64 *x = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *u = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    f64 dpar[2];
    i32 ipar[1] = {scale_param};
    i32 nr[2] = {4, nr2};
    bool vec[8] = {0};
    char note[80] = {0};
    i32 info;

    if (!e || !a || !y || !b || !x || !u || !dwork) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    switch (nr2) {
        case 1: dpar[0] = 1.5; dpar[1] = 1.5; break;
        case 2: dpar[0] = 0.5; dpar[1] = 1.5; break;
        case 3: dpar[0] = 10.0; dpar[1] = 0.0; break;
        case 4: dpar[0] = 1.5; dpar[1] = 0.0; break;
        default: dpar[0] = 1.5; dpar[1] = 1.5; break;
    }

    double times[BENCH_ITERATIONS];
    bench_stats_t stats;

    /* Warmup */
    for (int w = 0; w < BENCH_WARMUP; w++) {
        bb04ad("N", nr, dpar, ipar, vec, &n, &m,
               e, NMAX, a, NMAX, y, NMAX, b, MMAX,
               x, NMAX, u, NMAX, note, dwork, ldwork, &info);
    }

    /* Timed runs */
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        uint64_t start = bench_time_ns();
        bb04ad("N", nr, dpar, ipar, vec, &n, &m,
               e, NMAX, a, NMAX, y, NMAX, b, MMAX,
               x, NMAX, u, NMAX, note, dwork, ldwork, &info);
        uint64_t end = bench_time_ns();
        times[i] = bench_elapsed_us(start, end);
    }

    bench_compute_stats(times, BENCH_ITERATIONS, &stats);

    printf("bb04ad,ex4.%d,n=%d,%.2f,%.2f,%.2f,%.2f,%d\n",
           nr2, n, stats.mean_us, stats.min_us, stats.max_us, stats.stddev_us, info);

cleanup:
    free(e); free(a); free(y); free(b); free(x); free(u); free(dwork);
}

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
    printf("routine,example,size,mean_us,min_us,max_us,stddev_us,info\n");

    /* Group 4: Scalable parameter-dependent problems */
    for (int scale = 10; scale <= 50; scale += 10) {
        benchmark_bb04ad(1, scale);
        benchmark_bb04ad(2, scale);
        benchmark_bb04ad(3, scale);
        benchmark_bb04ad(4, scale);
    }

    return 0;
}
