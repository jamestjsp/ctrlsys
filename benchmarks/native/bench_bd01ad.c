/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Benchmark for BD01AD - CTDSX benchmark example generator (descriptor systems)
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

static void benchmark_bd01ad(int nr1, int nr2, int scale_param) {
    int n = NMAX, m = MMAX, p = PMAX;
    int ldwork = NMAX * (NMAX > 4 ? NMAX : 4);

    f64 *e = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *a = (f64 *)calloc(NMAX * NMAX, sizeof(f64));
    f64 *b = (f64 *)calloc(NMAX * MMAX, sizeof(f64));
    f64 *c = (f64 *)calloc(PMAX * NMAX, sizeof(f64));
    f64 *d = (f64 *)calloc(PMAX * MMAX, sizeof(f64));
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    f64 dpar[7] = {0};
    i32 ipar[7] = {0};
    i32 nr[2] = {nr1, nr2};
    bool vec[8] = {0};
    char chpar[256] = {0};
    i32 info;

    if (!e || !a || !b || !c || !d || !dwork) {
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
        bd01ad("D", nr, dpar, ipar, vec, &n, &m, &p,
               e, NMAX, a, NMAX, b, NMAX, c, PMAX, d, PMAX,
               chpar, dwork, ldwork, &info);
    }

    /* Timed runs */
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        uint64_t start = bench_time_ns();
        bd01ad("D", nr, dpar, ipar, vec, &n, &m, &p,
               e, NMAX, a, NMAX, b, NMAX, c, PMAX, d, PMAX,
               chpar, dwork, ldwork, &info);
        uint64_t end = bench_time_ns();
        times[i] = bench_elapsed_us(start, end);
    }

    bench_compute_stats(times, BENCH_ITERATIONS, &stats);

    printf("bd01ad,ex%d.%d,n=%d,%.2f,%.2f,%.2f,%.2f,%d\n",
           nr1, nr2, n, stats.mean_us, stats.min_us, stats.max_us, stats.stddev_us, info);

cleanup:
    free(e); free(a); free(b); free(c); free(d); free(dwork);
}

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
    printf("routine,example,size,mean_us,min_us,max_us,stddev_us,info\n");

    /* Group 1: Fixed-size examples */
    for (int ex = 1; ex <= 10; ex++) {
        benchmark_bd01ad(1, ex, 0);
    }

    /* Group 2: Parameter-dependent */
    for (int ex = 1; ex <= 6; ex++) {
        benchmark_bd01ad(2, ex, 0);
    }

    /* Group 3: Scalable - various sizes */
    for (int scale = 10; scale <= 50; scale += 10) {
        benchmark_bd01ad(3, 1, scale);
        benchmark_bd01ad(3, 2, scale);
    }

    return 0;
}
