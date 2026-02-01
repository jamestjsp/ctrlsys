/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Benchmark for SB02MD - Continuous-time Algebraic Riccati Equation solver
 *
 * Uses CAREX benchmark data from SLICOT-Reference/benchmark_data/BB01*.dat
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "slicot.h"
#include "../common/timing.h"
#include "../common/data_loader.h"

static void benchmark_sb02md(const char *filename) {
    bench_data_t data = {0};

    if (load_carex_data(filename, &data) < 0) {
        fprintf(stderr, "Failed to load: %s\n", filename);
        return;
    }

    int n = data.n;
    int n2 = 2 * n;
    int ldwork = 6 * n > 2 ? 6 * n : 2;

    /* Allocate working arrays */
    f64 *a_work = (f64 *)calloc(n * n, sizeof(f64));
    f64 *g = (f64 *)calloc(n * n, sizeof(f64));
    f64 *q = (f64 *)calloc(n * n, sizeof(f64));
    f64 *wr = (f64 *)calloc(n2, sizeof(f64));
    f64 *wi = (f64 *)calloc(n2, sizeof(f64));
    f64 *s = (f64 *)calloc(n2 * n2, sizeof(f64));
    f64 *u = (f64 *)calloc(n2 * n2, sizeof(f64));
    i32 *iwork = (i32 *)calloc(n2, sizeof(i32));
    f64 *dwork = (f64 *)calloc(ldwork, sizeof(f64));
    i32 *bwork = (i32 *)calloc(n2, sizeof(i32));

    if (!a_work || !g || !q || !wr || !wi || !s || !u || !iwork || !dwork || !bwork) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    /* For CAREX data: A is state matrix, B is input, C is Q weight matrix
     * We need to form G = B * B' for the standard Riccati form */
    int m = data.m;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            f64 sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += data.B[i + k * n] * data.B[j + k * n];
            }
            g[i + j * n] = sum;
        }
    }

    /* Timing arrays */
    double times[BENCH_ITERATIONS];
    bench_stats_t stats;
    f64 rcond;
    i32 info;

    /* Warmup runs */
    for (int w = 0; w < BENCH_WARMUP; w++) {
        memcpy(a_work, data.A, n * n * sizeof(f64));
        memcpy(q, data.C, n * n * sizeof(f64));
        sb02md("C", "D", "U", "N", "S", n,
               a_work, n, g, n, q, n,
               &rcond, wr, wi, s, n2, u, n2,
               iwork, dwork, ldwork, bwork, &info);
    }

    /* Timed runs */
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        memcpy(a_work, data.A, n * n * sizeof(f64));
        memcpy(q, data.C, n * n * sizeof(f64));

        uint64_t start = bench_time_ns();
        sb02md("C", "D", "U", "N", "S", n,
               a_work, n, g, n, q, n,
               &rcond, wr, wi, s, n2, u, n2,
               iwork, dwork, ldwork, bwork, &info);
        uint64_t end = bench_time_ns();

        times[i] = bench_elapsed_us(start, end);
    }

    bench_compute_stats(times, BENCH_ITERATIONS, &stats);

    /* Extract filename for output */
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;

    /* CSV output: routine,dataset,n,mean_us,min_us,max_us,stddev_us,info */
    printf("sb02md,%s,%d,%.2f,%.2f,%.2f,%.2f,%d\n",
           base, n, stats.mean_us, stats.min_us, stats.max_us, stats.stddev_us, info);

cleanup:
    free(a_work); free(g); free(q);
    free(wr); free(wi); free(s); free(u);
    free(iwork); free(dwork); free(bwork);
    free_bench_data(&data);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file.dat> [...]\n", argv[0]);
        fprintf(stderr, "       %s SLICOT-Reference/benchmark_data/BB01*.dat\n", argv[0]);
        return 1;
    }

    printf("routine,dataset,n,mean_us,min_us,max_us,stddev_us,info\n");

    for (int i = 1; i < argc; i++) {
        benchmark_sb02md(argv[i]);
    }

    return 0;
}
