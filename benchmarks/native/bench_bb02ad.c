/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Benchmark for BB02AD - DAREX benchmark example generator (discrete Riccati)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "slicot.h"
#include "../common/timing.h"

#define NMAX 100
#define MMAX 100
#define PMAX 100

static void benchmark_bb02ad(int nr1, int nr2, int scale_param) {
    int n = NMAX, m = MMAX, p = PMAX;
    int ldwork = NMAX * NMAX;

    f64 *a = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *b = (f64 *)calloc(NMAX * MMAX, sizeof(f64));
    f64 *c = (f64 *)calloc(PMAX * NMAX, sizeof(f64));
    f64 *q = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *r = (f64 *)calloc(MMAX * MMAX, sizeof(f64));
    f64 *s = (f64 *)calloc(NMAX * MMAX, sizeof(f64));
    f64 *x = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    f64 dpar[4] = {0};
    i32 ipar[3] = {0};
    i32 nr[2] = {nr1, nr2};
    bool bpar[7] = {true, true, false, true, true, false, false};
    bool vec[10] = {0};
    char chpar[256] = {0};
    i32 info;

    if (!a || !b || !c || !q || !r || !s || !x || !dwork) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    if (scale_param > 0) {
        ipar[0] = scale_param;
    }

    double times[BENCH_ITERATIONS];
    bench_stats_t stats;

    /* Warmup */
    for (int w = 0; w < BENCH_WARMUP; w++) {
        bb02ad("D", nr, dpar, ipar, bpar, chpar, vec, &n, &m, &p,
               a, NMAX, b, NMAX, c, PMAX, q, NMAX, r, MMAX, s, NMAX,
               x, NMAX, dwork, ldwork, &info);
    }

    /* Timed runs */
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        uint64_t start = bench_time_ns();
        bb02ad("D", nr, dpar, ipar, bpar, chpar, vec, &n, &m, &p,
               a, NMAX, b, NMAX, c, PMAX, q, NMAX, r, MMAX, s, NMAX,
               x, NMAX, dwork, ldwork, &info);
        uint64_t end = bench_time_ns();
        times[i] = bench_elapsed_us(start, end);
    }

    bench_compute_stats(times, BENCH_ITERATIONS, &stats);

    printf("bb02ad,ex%d.%d,n=%d,%.2f,%.2f,%.2f,%.2f,%d\n",
           nr1, nr2, n, stats.mean_us, stats.min_us, stats.max_us, stats.stddev_us, info);

cleanup:
    free(a); free(b); free(c); free(q); free(r); free(s); free(x); free(dwork);
}

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
    printf("routine,example,size,mean_us,min_us,max_us,stddev_us,info\n");

    /* Group 1: Fixed-size examples (1-13) */
    for (int ex = 1; ex <= 13; ex++) {
        benchmark_bb02ad(1, ex, 0);
    }

    /* Group 2: Parameter-dependent fixed size (1-5) */
    for (int ex = 1; ex <= 5; ex++) {
        benchmark_bb02ad(2, ex, 0);
    }

    /* Group 4: Scalable - various sizes */
    for (int scale = 10; scale <= 50; scale += 10) {
        benchmark_bb02ad(4, 1, scale);
    }

    return 0;
}
