/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Data loader for SLICOT benchmark .dat files
 */
#ifndef SLICOT_BENCHMARK_DATA_LOADER_H
#define SLICOT_BENCHMARK_DATA_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    int n;          /* State dimension */
    int m;          /* Number of inputs */
    int p;          /* Number of outputs */
    double *A;      /* n x n state matrix */
    double *B;      /* n x m input matrix */
    double *C;      /* p x n output matrix (or Q for CARE) */
    double *D;      /* Additional matrix (G for CARE, D for descriptor) */
    double *E;      /* Descriptor matrix (BD only) */
    double *X;      /* Solution matrix (if available) */
} bench_data_t;

/* Parse Fortran D-notation and implied exponent formats:
 * "1.234D+02" -> 123.4, "0.12317-02" -> 0.0012317 */
static inline double parse_fortran_double(const char *s) {
    char buf[64];
    int i = 0, j = 0;
    int has_exp = 0;

    while (s[i] && j < 63) {
        if (s[i] == 'D' || s[i] == 'd' || s[i] == 'E' || s[i] == 'e') {
            buf[j++] = 'E';
            has_exp = 1;
        } else if (!has_exp && j > 0 && (s[i] == '+' || s[i] == '-') &&
                   s[i-1] != 'E' && s[i-1] != 'e' && isdigit((unsigned char)s[i-1])) {
            buf[j++] = 'E';
            buf[j++] = s[i];
            has_exp = 1;
        } else {
            buf[j++] = s[i];
        }
        i++;
    }
    buf[j] = '\0';
    return atof(buf);
}

/* Read array of doubles from file, handling D-notation */
static inline int read_doubles(FILE *f, double *arr, int count) {
    char token[64];
    for (int i = 0; i < count; i++) {
        if (fscanf(f, "%63s", token) != 1) return -1;
        arr[i] = parse_fortran_double(token);
    }
    return 0;
}

/* Load CAREX benchmark data (BB01xxx.dat format)
 * Format: 4x4 A, 4x2 B, 4x4 Q (symmetric stored as full)
 * For larger examples, dimensions are inferred from file size
 */
static inline int load_carex_data(const char *filename, bench_data_t *data) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;

    /* Determine dimensions from filename pattern BB01XNN.dat
     * XNN encodes the example: 103=ex1.3, 104=ex1.4, etc. */
    int example_num = 0;
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    if (sscanf(base, "BB01%d.dat", &example_num) != 1 &&
        sscanf(base, "BB02%d.dat", &example_num) != 1) {
        fclose(f);
        return -1;
    }

    /* Common CAREX dimensions by example */
    int n = 4, m = 2;  /* defaults */
    if (example_num == 103) { n = 4; m = 2; }
    else if (example_num == 104) { n = 8; m = 2; }
    else if (example_num == 105) { n = 9; m = 3; }
    else if (example_num == 106) { n = 30; m = 3; }
    else if (example_num >= 2091) { n = 4; m = 2; }   /* Boeing 767 (file has extra data) */
    else if (example_num == 404) { n = 21; m = 1; }

    data->n = n;
    data->m = m;
    data->p = n;  /* For CARE, Q is n x n */

    data->A = (double *)calloc(n * n, sizeof(double));
    data->B = (double *)calloc(n * m, sizeof(double));
    data->C = (double *)calloc(n * n, sizeof(double));  /* Q matrix */
    data->D = NULL;
    data->E = NULL;
    data->X = NULL;

    if (!data->A || !data->B || !data->C) {
        fclose(f);
        return -1;
    }

    /* Read A (n x n) */
    if (read_doubles(f, data->A, n * n) < 0) goto err;

    /* Read B (n x m) */
    if (read_doubles(f, data->B, n * m) < 0) goto err;

    /* Read Q (n x n) */
    if (read_doubles(f, data->C, n * n) < 0) {
        /* Q might not be present for all examples */
        memset(data->C, 0, n * n * sizeof(double));
    }

    fclose(f);
    return 0;

err:
    free(data->A); free(data->B); free(data->C);
    data->A = data->B = data->C = NULL;
    fclose(f);
    return -1;
}

/* Load DAREX benchmark data (BB02xxx.dat format)
 * Format: A, B, Q matrices for discrete-time Riccati
 */
static inline int load_darex_data(const char *filename, bench_data_t *data) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;

    int example_num = 0;
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    if (sscanf(base, "BB02%d.dat", &example_num) != 1) {
        fclose(f);
        return -1;
    }

    /* DAREX dimensions by example
     * Extracted from SLICOT Working Note 1998-9: DAREX benchmark library */
    int n = 4, m = 2;
    if (example_num == 105) { n = 4; m = 2; }       /* BB02105: n=4, m=2 */
    else if (example_num == 106) { n = 4; m = 2; }  /* BB02106: n=4, m=2 */
    else if (example_num == 107) { n = 4; m = 2; }  /* BB02107: n=4, m=2 */
    else if (example_num == 108) { n = 4; m = 2; }  /* BB02108: n=4, m=2 */
    else if (example_num == 110) { n = 4; m = 2; }  /* BB02110: n=4, m=2 */
    else if (example_num == 111) { n = 4; m = 1; }  /* BB02111: n=4, m=1 */
    else if (example_num == 113) { n = 5; m = 2; }  /* BB02113: n=5, m=2 */

    data->n = n;
    data->m = m;
    data->p = n;

    data->A = (double *)calloc(n * n, sizeof(double));
    data->B = (double *)calloc(n * m, sizeof(double));
    data->C = (double *)calloc(n * n, sizeof(double));
    data->D = NULL;
    data->E = NULL;
    data->X = NULL;

    if (!data->A || !data->B || !data->C) {
        fclose(f);
        return -1;
    }

    if (read_doubles(f, data->A, n * n) < 0) goto err;
    if (read_doubles(f, data->B, n * m) < 0) goto err;
    if (read_doubles(f, data->C, n * n) < 0) {
        memset(data->C, 0, n * n * sizeof(double));
    }

    fclose(f);
    return 0;

err:
    free(data->A); free(data->B); free(data->C);
    data->A = data->B = data->C = NULL;
    fclose(f);
    return -1;
}

/* Load descriptor system benchmark data (BD01xxx.dat format) */
static inline int load_ctdsx_data(const char *filename, bench_data_t *data) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;

    int example_num = 0;
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    if (sscanf(base, "BD01%d.dat", &example_num) != 1 &&
        sscanf(base, "BD02%d.dat", &example_num) != 1) {
        fclose(f);
        return -1;
    }

    /* CTDSX dimensions by example */
    int n = 4, m = 2, p = 2;
    if (example_num == 103) { n = 4; m = 2; p = 2; }
    else if (example_num == 104) { n = 8; m = 2; p = 2; }
    else if (example_num == 105) { n = 9; m = 3; p = 9; }
    else if (example_num == 106) { n = 30; m = 3; p = 5; }
    else if (example_num == 107) { n = 6; m = 1; p = 1; }
    else if (example_num == 108) { n = 6; m = 1; p = 6; }
    else if (example_num == 109) { n = 55; m = 4; p = 1; }
    else if (example_num == 110) { n = 8; m = 2; p = 2; }
    else if (example_num == 203) { n = 8; m = 2; p = 2; }
    else if (example_num == 206) { n = 5; m = 1; p = 1; }
    else if (example_num == 304) { n = 4; m = 21; p = 21; }
    else if (example_num == 2051) { n = 2; m = 1; p = 1; }
    else if (example_num == 2052) { n = 4; m = 1; p = 1; }
    else if (example_num == 2053) { n = 6; m = 1; p = 1; }
    else if (example_num == 2054) { n = 8; m = 1; p = 1; }
    else if (example_num == 2055) { n = 10; m = 1; p = 1; }
    else if (example_num == 2056) { n = 12; m = 1; p = 1; }
    else if (example_num == 2057) { n = 20; m = 1; p = 1; }

    data->n = n;
    data->m = m;
    data->p = p;

    data->A = (double *)calloc(n * n, sizeof(double));
    data->B = (double *)calloc(n * m, sizeof(double));
    data->C = (double *)calloc(p * n, sizeof(double));
    data->D = NULL;
    data->E = NULL;
    data->X = NULL;

    if (!data->A || !data->B || !data->C) {
        fclose(f);
        return -1;
    }

    /* Read A (n x n) */
    if (read_doubles(f, data->A, n * n) < 0) goto err;

    /* Read B (n x m) */
    if (read_doubles(f, data->B, n * m) < 0) goto err;

    /* Try to read C (p x n) - not always present */
    if (read_doubles(f, data->C, p * n) < 0) {
        memset(data->C, 0, p * n * sizeof(double));
    }

    fclose(f);
    return 0;

err:
    free(data->A); free(data->B); free(data->C);
    data->A = data->B = data->C = NULL;
    fclose(f);
    return -1;
}

/* Load DTDSX benchmark data (BD02xxx.dat format)
 * Format: A, B, C matrices for discrete-time descriptor systems
 */
static inline int load_dtdsx_data(const char *filename, bench_data_t *data) {
    FILE *f = fopen(filename, "r");
    if (!f) return -1;

    int example_num = 0;
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    if (sscanf(base, "BD02%d.dat", &example_num) != 1) {
        fclose(f);
        return -1;
    }

    /* DTDSX dimensions by example */
    int n = 4, m = 2, p = 2;
    if (example_num == 106) { n = 4; m = 2; p = 2; }
    else if (example_num == 107) { n = 4; m = 2; p = 2; }
    else if (example_num == 108) { n = 4; m = 2; p = 4; }
    else if (example_num == 109) { n = 5; m = 1; p = 1; }
    else if (example_num == 111) { n = 9; m = 3; p = 9; }
    else if (example_num == 112) { n = 3; m = 1; p = 1; }

    data->n = n;
    data->m = m;
    data->p = p;

    data->A = (double *)calloc(n * n, sizeof(double));
    data->B = (double *)calloc(n * m, sizeof(double));
    data->C = (double *)calloc(p * n, sizeof(double));
    data->D = NULL;
    data->E = NULL;
    data->X = NULL;

    if (!data->A || !data->B || !data->C) {
        fclose(f);
        return -1;
    }

    if (read_doubles(f, data->A, n * n) < 0) goto err;
    if (read_doubles(f, data->B, n * m) < 0) goto err;
    if (read_doubles(f, data->C, p * n) < 0) {
        memset(data->C, 0, p * n * sizeof(double));
    }

    fclose(f);
    return 0;

err:
    free(data->A); free(data->B); free(data->C);
    data->A = data->B = data->C = NULL;
    fclose(f);
    return -1;
}

static inline void free_bench_data(bench_data_t *data) {
    if (data->A) free(data->A);
    if (data->B) free(data->B);
    if (data->C) free(data->C);
    if (data->D) free(data->D);
    if (data->E) free(data->E);
    if (data->X) free(data->X);
    memset(data, 0, sizeof(*data));
}

#endif /* SLICOT_BENCHMARK_DATA_LOADER_H */
