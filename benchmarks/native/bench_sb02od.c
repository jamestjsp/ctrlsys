/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Benchmark for SB02OD - Discrete-time Algebraic Riccati Equation solver
 *
 * Uses DAREX benchmark data from SLICOT-Reference/benchmark_data/BB02*.dat
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slicot.h"
#include "../common/timing.h"
#include "../common/data_loader.h"

static void benchmark_sb02od(const char *filename) {
    bench_data_t data = {0};

    if (load_darex_data(filename, &data) < 0) {
        fprintf(stderr, "Failed to load: %s\n", filename);
        return;
    }

    int n = data.n;
    int m = data.m;
    int n2 = 2 * n;
    int n2m = 2 * n + m;
    int ldwork = 7 * (2 * n + 1) + 16;
    if (ldwork < 16 * n) ldwork = 16 * n;
    if (ldwork < 2 * n + m) ldwork = 2 * n + m;
    if (ldwork < 3 * m) ldwork = 3 * m;

    f64 *a_work = (f64 *)calloc(n * n, sizeof(f64));
    f64 *b_work = (f64 *)calloc(n * m, sizeof(f64));
    f64 *q = (f64 *)calloc(n * n, sizeof(f64));
    f64 *r = (f64 *)calloc(m * m, sizeof(f64));
    f64 *l = (f64 *)calloc(n * m, sizeof(f64));
    f64 *x = (f64 *)calloc(n * n, sizeof(f64));
    f64 *alfar = (f64 *)calloc(n2, sizeof(f64));
    f64 *alfai = (f64 *)calloc(n2, sizeof(f64));
    f64 *beta = (f64 *)calloc(n2, sizeof(f64));
    f64 *s = (f64 *)calloc(n2m * n2m, sizeof(f64));
    f64 *t = (f64 *)calloc(n2m * n2m, sizeof(f64));
    f64 *u = (f64 *)calloc(n2 * n2, sizeof(f64));
    i32 *iwork = (i32 *)calloc(n2m > m ? n2m : m, sizeof(i32));
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));

    if (!a_work || !b_work || !q || !r || !l || !x ||
        !alfar || !alfai || !beta || !s || !t || !u || !iwork || !dwork) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    /* Initialize R as identity (standard case) */
    for (int i = 0; i < m; i++) {
        r[i + i * m] = 1.0;
    }

    double times[BENCH_ITERATIONS];
    bench_stats_t stats;
    f64 rcond;
    i32 info;

    /* Warmup runs */
    for (int w = 0; w < BENCH_WARMUP; w++) {
        memcpy(a_work, data.A, n * n * sizeof(f64));
        memcpy(b_work, data.B, n * m * sizeof(f64));
        memcpy(q, data.C, n * n * sizeof(f64));
        for (int i = 0; i < m; i++) {
            r[i + i * m] = 1.0;
        }

        sb02od("D", "B", "N", "U", "Z", "S",
               n, m, 0,
               a_work, n, b_work, n, q, n, r, m, l, n,
               &rcond, x, n,
               alfar, alfai, beta,
               s, n2m, t, n2m, u, n2,
               0.0, iwork, dwork, ldwork, &info);
    }

    /* Timed runs */
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        memcpy(a_work, data.A, n * n * sizeof(f64));
        memcpy(b_work, data.B, n * m * sizeof(f64));
        memcpy(q, data.C, n * n * sizeof(f64));
        for (int j = 0; j < m; j++) {
            r[j + j * m] = 1.0;
        }

        uint64_t start = bench_time_ns();
        sb02od("D", "B", "N", "U", "Z", "S",
               n, m, 0,
               a_work, n, b_work, n, q, n, r, m, l, n,
               &rcond, x, n,
               alfar, alfai, beta,
               s, n2m, t, n2m, u, n2,
               0.0, iwork, dwork, ldwork, &info);
        uint64_t end = bench_time_ns();

        times[i] = bench_elapsed_us(start, end);
    }

    bench_compute_stats(times, BENCH_ITERATIONS, &stats);

    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;

    printf("sb02od,%s,%d,%.2f,%.2f,%.2f,%.2f,%d\n",
           base, n, stats.mean_us, stats.min_us, stats.max_us, stats.stddev_us, info);

cleanup:
    free(a_work); free(b_work); free(q); free(r); free(l); free(x);
    free(alfar); free(alfai); free(beta); free(s); free(t); free(u);
    free(iwork); free(dwork);
    free_bench_data(&data);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file.dat> [...]\n", argv[0]);
        fprintf(stderr, "       %s SLICOT-Reference/benchmark_data/BB02*.dat\n", argv[0]);
        return 1;
    }

    printf("routine,dataset,n,mean_us,min_us,max_us,stddev_us,info\n");

    for (int i = 1; i < argc; i++) {
        benchmark_sb02od(argv[i]);
    }

    return 0;
}
