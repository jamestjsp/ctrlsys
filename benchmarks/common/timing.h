/*
 * SPDX-License-Identifier: BSD-3-Clause
 * High-resolution timing utilities for benchmarks
 */
#ifndef SLICOT_BENCHMARK_TIMING_H
#define SLICOT_BENCHMARK_TIMING_H

#include <time.h>
#include <stdint.h>
#include <math.h>

#ifdef __MACH__
#include <mach/mach_time.h>
#endif

typedef struct {
    double min_us;
    double max_us;
    double mean_us;
    double stddev_us;
    int iterations;
} bench_stats_t;

static inline uint64_t bench_time_ns(void) {
#ifdef __MACH__
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

static inline double bench_elapsed_us(uint64_t start, uint64_t end) {
    return (double)(end - start) / 1000.0;
}

#define BENCH_WARMUP 3
#define BENCH_ITERATIONS 10

#define BENCH_START() uint64_t _bench_start = bench_time_ns()
#define BENCH_END() bench_time_ns()
#define BENCH_ELAPSED_US(start, end) bench_elapsed_us(start, end)

static inline void bench_compute_stats(double *times, int n, bench_stats_t *stats) {
    if (n <= 0) return;

    stats->iterations = n;
    stats->min_us = times[0];
    stats->max_us = times[0];
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        if (times[i] < stats->min_us) stats->min_us = times[i];
        if (times[i] > stats->max_us) stats->max_us = times[i];
        sum += times[i];
    }
    stats->mean_us = sum / n;

    double var_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = times[i] - stats->mean_us;
        var_sum += d * d;
    }
    stats->stddev_us = (n > 1) ? sqrt(var_sum / (n - 1)) : 0.0;
}

#endif /* SLICOT_BENCHMARK_TIMING_H */
