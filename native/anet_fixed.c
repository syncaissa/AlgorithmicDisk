/*
 * N6 — compiled fixed-point AlgorithmicNET + range coder (C, OpenMP).
 * Integer-only engine (mirrors fixedpoint.py exactly) and a carry-less 32-bit range coder that
 * mirrors predictive.py's RangeEncoder/RangeDecoder byte-for-byte, so Seeds are interchangeable
 * between the Python and C implementations.  Blocks are independent -> OpenMP over blocks.
 *
 *   gcc -O3 -march=native -fopenmp -shared -fPIC anet_fixed.c -o libanet_fixed.so
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define QW 12
#define QA 15
#define TANH_N 4096
#define TANH_RANGE 8
#define START 256
#define TOTAL (1 << 16)
#define TOP (1u << 24)
#define BOT (1u << 16)

typedef struct {
    int hidden, emb;
    const int32_t *E, *W_x, *W_h, *b_h, *W_o, *b_o;   /* E: (257,emb) W_x: (hidden,emb) W_h: (hidden,hidden) W_o: (256,hidden) */
    const int64_t *tanh_lut;                          /* 4096 */
    const int64_t *neg2_lut;                          /* 256 */
} engine_t;

/* ---------------------------------------------------------------- engine step (int64 accumulators) */
static inline __attribute__((always_inline)) int64_t dot(const int32_t *restrict a, const int32_t *restrict b, int n) {
    int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int j = 0; j + 3 < n; j += 4) {                       /* four independent int64 accumulators; int32 x int32 -> int64 exact */
        s0 += (int64_t)a[j] * b[j]; s1 += (int64_t)a[j + 1] * b[j + 1];
        s2 += (int64_t)a[j + 2] * b[j + 2]; s3 += (int64_t)a[j + 3] * b[j + 3];
    }
    return s0 + s1 + s2 + s3;                                  /* hidden is a multiple of 4 */
}
/* xe[s*hidden + i] = (W_x · E[s]) >> (2*QW-QA)  precomputed once per engine (exactly what the Python does per step) */
static void step(const engine_t *e, const int64_t *xe, int32_t *H, int prev, int64_t *logit, int32_t *tmp) {
    const int hid = e->hidden;
    const int64_t *x = xe + (size_t)prev * hid;
    for (int i = 0; i < hid; i++) {
        int64_t acc = dot(e->W_h + (size_t)i * hid, H, hid);
        int64_t u = (acc >> QW) + x[i] + ((int64_t)e->b_h[i] << (QA - QW));
        int64_t idx = ((u + ((int64_t)TANH_RANGE << QA)) * TANH_N) >> (QA + 4);
        if (idx < 0) idx = 0; if (idx > TANH_N - 1) idx = TANH_N - 1;
        tmp[i] = (int32_t)e->tanh_lut[idx];
    }
    memcpy(H, tmp, sizeof(int32_t) * hid);
    for (int k = 0; k < 256; k++)
        logit[k] = (dot(e->W_o + (size_t)k * hid, H, hid) >> QW) + ((int64_t)e->b_o[k] << (QA - QW));
}
static int64_t *make_xe(const engine_t *e) {
    int64_t *xe = malloc(sizeof(int64_t) * 257 * e->hidden);
    for (int s = 0; s < 257; s++) for (int i = 0; i < e->hidden; i++) {
        int64_t ax = 0; for (int j = 0; j < e->emb; j++) ax += (int64_t)e->W_x[(size_t)i * e->emb + j] * e->E[(size_t)s * e->emb + j];
        xe[(size_t)s * e->hidden + i] = ax >> (2 * QW - QA);
    }
    return xe;
}

/* integer soft-max -> cumulative table cum[0..256], cum[256] == TOTAL  (mirrors FixedPointANET.freqs) */
static void freqs(const engine_t *e, const int64_t *logit, int64_t *cum) {
    int64_t mx = logit[0]; for (int k = 1; k < 256; k++) if (logit[k] > mx) mx = logit[k];
    int64_t p[256], s = 0;
    for (int k = 0; k < 256; k++) {
        int64_t z = logit[k] - mx;                       /* <= 0 */
        int64_t q = ((-z) * 47274) >> 22;                /* -z/ln2 in Q8 */
        int64_t ip = q >> 8, fp = q & 255;
        p[k] = (ip >= 62) ? 0 : (e->neg2_lut[fp] >> (ip < 62 ? ip : 62));
        s += p[k];
    }
    if (s == 0) s = 1;
    int64_t f[256], tot = 0, amax = 0;
    for (int k = 0; k < 256; k++) { f[k] = (p[k] * (TOTAL - 256)) / s + 1; tot += f[k]; if (f[k] > f[amax]) amax = k; }
    f[amax] += TOTAL - tot;
    cum[0] = 0; for (int k = 0; k < 256; k++) cum[k + 1] = cum[k] + f[k];
}

/* ---------------------------------------------------------------- range coder (mirrors predictive.py) */
typedef struct { uint64_t low; uint64_t range; uint8_t *out; size_t n, cap; } enc_t;
static void enc_init(enc_t *c, uint8_t *buf, size_t cap) { c->low = 0; c->range = 0xFFFFFFFFu; c->out = buf; c->n = 0; c->cap = cap; }
static void enc_put(enc_t *c, uint8_t b) { if (c->n < c->cap) c->out[c->n] = b; c->n++; }
static void enc_encode(enc_t *c, uint64_t cum, uint64_t freq, uint64_t total) {
    uint64_t r = c->range / total;
    c->low += r * cum; c->range = r * freq;
    for (;;) {
        if (((c->low ^ (c->low + c->range)) & 0xFFFFFFFFull) < TOP && ((c->low ^ (c->low + c->range)) >> 32) == 0) { }
        else if (c->range < BOT) { c->range = ((~c->low) + 1) & (BOT - 1); }
        else break;
        enc_put(c, (uint8_t)((c->low >> 24) & 0xFF));
        c->low = (c->low << 8) & 0xFFFFFFFFull; c->range = (c->range << 8) & 0xFFFFFFFFull;
    }
}
static size_t enc_finish(enc_t *c) { for (int i = 0; i < 4; i++) { enc_put(c, (uint8_t)((c->low >> 24) & 0xFF)); c->low = (c->low << 8) & 0xFFFFFFFFull; } return c->n; }

typedef struct { const uint8_t *data; size_t n, pos; uint64_t low, range, code, r; } dec_t;
static uint8_t dec_byte(dec_t *d) { return d->pos < d->n ? d->data[d->pos++] : (d->pos++, 0); }
static void dec_init(dec_t *d, const uint8_t *data, size_t n) { d->data = data; d->n = n; d->pos = 0; d->low = 0; d->range = 0xFFFFFFFFu; d->code = 0; for (int i = 0; i < 4; i++) d->code = ((d->code << 8) | dec_byte(d)) & 0xFFFFFFFFull; }
static uint64_t dec_target(dec_t *d, uint64_t total) { d->r = d->range / total; uint64_t t = (d->code - d->low) / d->r; return t < total - 1 ? t : total - 1; }
static void dec_decode(dec_t *d, uint64_t cum, uint64_t freq) {
    d->low += d->r * cum; d->range = d->r * freq;
    for (;;) {
        if (((d->low ^ (d->low + d->range)) & 0xFFFFFFFFull) < TOP && ((d->low ^ (d->low + d->range)) >> 32) == 0) { }
        else if (d->range < BOT) { d->range = ((~d->low) + 1) & (BOT - 1); }
        else break;
        d->code = ((d->code << 8) | dec_byte(d)) & 0xFFFFFFFFull;
        d->low = (d->low << 8) & 0xFFFFFFFFull; d->range = (d->range << 8) & 0xFFFFFFFFull;
    }
}

/* ---------------------------------------------------------------- public API: per-block streams */
static engine_t mk(int hidden, int emb, const int32_t *E, const int32_t *W_x, const int32_t *W_h, const int32_t *b_h,
                   const int32_t *W_o, const int32_t *b_o, const int64_t *tanh_lut, const int64_t *neg2_lut) {
    engine_t e = { hidden, emb, E, W_x, W_h, b_h, W_o, b_o, tanh_lut, neg2_lut }; return e;
}

/* encode: data[n] in blocks of L; out[i*cap .. ] receives stream i, out_len[i] its length. returns #blocks */
int encode_blocks(int hidden, int emb, const int32_t *E, const int32_t *W_x, const int32_t *W_h, const int32_t *b_h,
                  const int32_t *W_o, const int32_t *b_o, const int64_t *tanh_lut, const int64_t *neg2_lut,
                  const uint8_t *data, int64_t n, int L, uint8_t *out, int64_t cap, int64_t *out_len, int nthreads) {
    engine_t e = mk(hidden, emb, E, W_x, W_h, b_h, W_o, b_o, tanh_lut, neg2_lut);
    int64_t *xe = make_xe(&e);
    int nb = (int)((n + L - 1) / L);
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#endif
#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < nb; j++) {
        int32_t *H = calloc(hidden, sizeof(int32_t)); int32_t *tmp = malloc(sizeof(int32_t) * hidden);
        int64_t logit[256], cum[257]; enc_t c; enc_init(&c, out + (size_t)j * cap, cap);
        int prev = START; int64_t b0 = (int64_t)j * L, b1 = b0 + L < n ? b0 + L : n;
        for (int64_t i = b0; i < b1; i++) {
            step(&e, xe, H, prev, logit, tmp); freqs(&e, logit, cum);
            int ch = data[i]; enc_encode(&c, cum[ch], cum[ch + 1] - cum[ch], TOTAL); prev = ch;
        }
        out_len[j] = (int64_t)enc_finish(&c); free(H); free(tmp);
    }
    free(xe);
    return nb;
}

/* decode: streams concatenated in `seed` with offsets off[j], len[j]; writes data[n] */
int decode_blocks(int hidden, int emb, const int32_t *E, const int32_t *W_x, const int32_t *W_h, const int32_t *b_h,
                  const int32_t *W_o, const int32_t *b_o, const int64_t *tanh_lut, const int64_t *neg2_lut,
                  const uint8_t *seed, const int64_t *off, const int64_t *len, int nb, uint8_t *data, int64_t n, int L, int nthreads) {
    engine_t e = mk(hidden, emb, E, W_x, W_h, b_h, W_o, b_o, tanh_lut, neg2_lut);
    int64_t *xe = make_xe(&e);
#ifdef _OPENMP
    if (nthreads > 0) omp_set_num_threads(nthreads);
#endif
#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < nb; j++) {
        int32_t *H = calloc(hidden, sizeof(int32_t)); int32_t *tmp = malloc(sizeof(int32_t) * hidden);
        int64_t logit[256], cum[257]; dec_t d; dec_init(&d, seed + off[j], (size_t)len[j]);
        int prev = START; int64_t b0 = (int64_t)j * L, b1 = b0 + L < n ? b0 + L : n;
        for (int64_t i = b0; i < b1; i++) {
            step(&e, xe, H, prev, logit, tmp); freqs(&e, logit, cum);
            uint64_t t = dec_target(&d, TOTAL); int ch = 0;
            { int lo = 0, hi = 255; while (lo < hi) { int mid = (lo + hi + 1) / 2; if ((uint64_t)cum[mid] <= t) lo = mid; else hi = mid - 1; } ch = lo; }
            dec_decode(&d, cum[ch], cum[ch + 1] - cum[ch]); data[i] = (uint8_t)ch; prev = ch;
        }
        free(H); free(tmp);
    }
    free(xe);
    return nb;
}
