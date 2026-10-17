#pragma once

// (ticket P4h) the MTP draft's pick of its next token from its candidates, on the device (GGML_OP_DRAFT_PICK), so the
// draft's steps chain without the host. One row is one draft step of one sequence: k candidates (logits, token ids and
// the rows of the draft vocabulary subset they came from), in the order the device top-k gives them.
//
// The pick reproduces the host's draft loop (the old engine's common/speculative.cpp draft(), removed in FL2):
//   - the candidates in the order the draft's own sampler leaves them: logits descending (ties: earlier first), and p0,
//     its probability of the first one (the top-k + dist chain: expf(l - l0) summed in float, in that order);
//   - mode 0: the first candidate (the argmax; no q row);
//   - mode 1: [TAG_SPEC_REJECTION] the old rejection draft sampler on those candidates: the target's sampler chain but its
//     dist (logit bias, top-k, top-p, min-p and temperature, in the chain's order; the chain's other samplers are neutral
//     for the requests this runs for, the old device chain), LLAMA_SPEC_REJECTION_TEMP, the probabilities
//     (expf in float, the sum in double), and the inverse CDF of the draw u (drawn on the host from the target's stream).
//     When the chain leaves no candidate the first candidate is picked with no q row, as on the host.
// The q row and q(token) are the device's own: the rejection step reads them, so it stays exact in distribution whatever
// the last bits of expf are on either side.
//
// params, float [GGML_DRAFT_PICK_NP] per row:
//   [0] mode (0: argmax, 1: the chain), [1] top_k, [2] top_p, [3] min_p, [4] temp, [5] min_keep, [6] u, [7] n_bias,
//   [8 .. 8 + 2*GGML_DRAFT_PICK_NBIAS) the logit bias as (token id, bias) pairs, [24 .. 28) the chain's order as codes
//   (1: top_k, 2: top_p, 3: min_p, 4: temp; 0 ends it), [28] LLAMA_SPEC_REJECTION_TEMP (0: none), [29] the rest of u
//   (u is the host's double draw as [6] + [29], two floats)
// output, int32 [GGML_DRAFT_PICK_HDR + 2*k] per row:
//   [0] token id, [1] its subset row, [2] q(token) as float bits, [3] p0 as float bits, [4] n_q, [5] the pick's index among
//   the candidates as given, [6] 1 when the chain left no candidate, [7] 0, then the q row: n_q token ids, and from
//   [GGML_DRAFT_PICK_HDR + k] their probabilities as float bits

#include <math.h>
#include <stdint.h>
#include <string.h>

#define GGML_DRAFT_PICK_K_MAX 64
#define GGML_DRAFT_PICK_NP    32
#define GGML_DRAFT_PICK_NBIAS 8
#define GGML_DRAFT_PICK_HDR   8

#if defined(__CUDACC__)
#define GGML_DRAFT_PICK_FN static __device__ __forceinline__
// the CUDA backend builds with --use_fast_math: exp and log in double (rounded once to float, which is glibc's expf and
// logf in all but the rarest cases) and IEEE float division, so the device follows the host's arithmetic
#define GGML_DRAFT_PICK_EXPF(x)    ((float) exp((double) (x)))
#define GGML_DRAFT_PICK_LOGF(x)    ((float) log((double) (x)))
#define GGML_DRAFT_PICK_DIVF(a, b) __fdiv_rn((a), (b))
#else
#define GGML_DRAFT_PICK_FN static inline
#define GGML_DRAFT_PICK_EXPF(x)    expf(x)
#define GGML_DRAFT_PICK_LOGF(x)    logf(x)
#define GGML_DRAFT_PICK_DIVF(a, b) ((a) / (b))
#endif

GGML_DRAFT_PICK_FN int32_t ggml_draft_pick_f2i(float f) {
    int32_t i;
    memcpy(&i, &f, sizeof(i));
    return i;
}

GGML_DRAFT_PICK_FN int ggml_draft_pick_isfinite(float f) {
    return !(f != f) && f != INFINITY && f != -INFINITY;
}

// stable sort of the first n of (l, e) by l descending
GGML_DRAFT_PICK_FN void ggml_draft_pick_sort(float * l, int * e, float * p, int n) {
    for (int i = 1; i < n; ++i) {
        const float li = l[i];
        const int   ei = e[i];
        const float pi = p[i];
        int j = i - 1;
        while (j >= 0 && l[j] < li) {
            l[j + 1] = l[j];
            e[j + 1] = e[j];
            p[j + 1] = p[j];
            --j;
        }
        l[j + 1] = li;
        e[j + 1] = ei;
        p[j + 1] = pi;
    }
}

// the k candidates of a row from its source (src_mode, ggml.h ggml_draft_pick): logits, token ids and subset rows. src_mode 1
// takes the k highest of the n_a/2 gathered values, ties by position, in that order
GGML_DRAFT_PICK_FN void ggml_draft_pick_cands(const int src_mode, const int k, const int n_a,
        const float * a, const int32_t * b, const int32_t * c, float * logits, int32_t * ids, int32_t * rows) {
    if (src_mode == 0) {
        for (int i = 0; i < k; ++i) {
            logits[i] = a[i];
            ids[i]    = b[i];
            rows[i]   = c[i];
        }
    } else if (src_mode == 1) {
        const int n_c = n_a/2;
        for (int i = 0; i < n_c; ++i) {
            int r = 0;
            for (int j = 0; j < n_c; ++j) {
                r += a[j] > a[i] || (a[j] == a[i] && j < i);
            }
            if (r < k) {
                logits[r] = a[i];
                rows[r]   = (int32_t) a[n_c + i];
                ids[r]    = c[rows[r]];
            }
        }
    } else {
        for (int i = 0; i < k; ++i) {
            const int32_t r = b[i];
            logits[i] = a[r];
            rows[i]   = r;
            ids[i]    = c[r];
        }
    }
}

GGML_DRAFT_PICK_FN void ggml_draft_pick_row(
        const float * logits, const int32_t * ids, const int32_t * rows, const int k, const float * prm, int32_t * out) {
    float l[GGML_DRAFT_PICK_K_MAX];
    float p[GGML_DRAFT_PICK_K_MAX];
    int   e[GGML_DRAFT_PICK_K_MAX]; // index into the candidates as given

    for (int i = 0; i < GGML_DRAFT_PICK_HDR + 2*k; ++i) {
        out[i] = 0;
    }

    // the draft's own order and p0
    for (int i = 0; i < k; ++i) {
        l[i] = logits[i];
        e[i] = i;
        p[i] = 0.0f;
    }
    ggml_draft_pick_sort(l, e, p, k);
    {
        float cum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cum += GGML_DRAFT_PICK_EXPF(l[i] - l[0]);
        }
        const float p0 = GGML_DRAFT_PICK_DIVF(GGML_DRAFT_PICK_EXPF(l[0] - l[0]), cum);
        out[3] = ggml_draft_pick_f2i(p0);
    }

    const int mode = (int) prm[0];
    int  sel = 0;   // into the arrays
    int  n   = k;   // candidates left
    bool none = mode != 1;
    if (mode == 1) {
        bool sorted = false; // the old rejection draft sampler handed the chain an array it marks unsorted

        // logit bias (first in the chain)
        const int n_bias = (int) prm[7];
        for (int i = 0; i < n; ++i) {
            for (int b = 0; b < n_bias && b < GGML_DRAFT_PICK_NBIAS; ++b) {
                if (ids[e[i]] == (int32_t) prm[8 + 2*b]) {
                    l[i] += prm[8 + 2*b + 1];
                    break;
                }
            }
        }

        const float min_keep = prm[5];
        for (int o = 0; o < 4; ++o) {
            const int code = (int) prm[24 + o];
            if (code == 0) {
                break;
            }
            if (code == 1) {
                // top_k
                int kk = (int) prm[1];
                if (kk <= 0) {
                    continue;
                }
                kk = kk < n ? kk : n;
                if (!sorted) {
                    ggml_draft_pick_sort(l, e, p, n);
                    sorted = true;
                }
                n = kk;
            } else if (code == 2) {
                // top_p
                const float tp = prm[2];
                if (tp >= 1.0f) {
                    continue;
                }
                // softmax in the array's order (llama_sampler_softmax_impl, no sort)
                float max_l = l[0];
                if (!sorted) {
                    for (int i = 1; i < n; ++i) {
                        max_l = l[i] > max_l ? l[i] : max_l;
                    }
                }
                float cum = 0.0f;
                for (int i = 0; i < n; ++i) {
                    p[i] = GGML_DRAFT_PICK_EXPF(l[i] - max_l);
                    cum += p[i];
                }
                for (int i = 0; i < n; ++i) {
                    p[i] = GGML_DRAFT_PICK_DIVF(p[i], cum);
                }
                if (!sorted) {
                    ggml_draft_pick_sort(l, e, p, n);
                    sorted = true;
                }
                float cs = 0.0f;
                int last = n;
                for (int i = 0; i < n; ++i) {
                    cs += p[i];
                    if (cs >= tp && (float) (i + 1) >= min_keep) {
                        last = i + 1;
                        break;
                    }
                }
                n = last;
            } else if (code == 3) {
                // min_p
                const float mp = prm[3];
                if (mp <= 0.0f || n == 0) {
                    continue;
                }
                bool applied = false;
                if (!sorted) {
                    float max_l = -3.402823466e+38f;
                    for (int i = 0; i < n; ++i) {
                        max_l = l[i] > max_l ? l[i] : max_l;
                    }
                    const float min_l = max_l + GGML_DRAFT_PICK_LOGF(mp);
                    int nf = 0;
                    for (int i = 0; i < n; ++i) {
                        nf += l[i] >= min_l;
                    }
                    // as the host: the filtered tokens replace the array only when there are enough of them
                    if (nf > 0 && (float) nf >= min_keep) {
                        nf = 0;
                        for (int i = 0; i < n; ++i) {
                            if (l[i] >= min_l) {
                                l[nf] = l[i];
                                e[nf] = e[i];
                                p[nf] = p[i];
                                nf++;
                            }
                        }
                        n = nf;
                        applied = true;
                    }
                }
                if (!applied) {
                    if (!sorted) {
                        ggml_draft_pick_sort(l, e, p, n);
                        sorted = true;
                    }
                    const float min_l = l[0] + GGML_DRAFT_PICK_LOGF(mp);
                    int i = 1;
                    for (; i < n; ++i) {
                        if (l[i] < min_l && (float) i >= min_keep) {
                            break;
                        }
                    }
                    n = i;
                }
            } else if (code == 4) {
                // temperature (no dynamic range)
                const float t = prm[4];
                if (n == 0) {
                    continue;
                }
                if (t <= 0.0f) {
                    int   max_i = 0;
                    float max_l = l[0];
                    for (int i = 1; i < n; ++i) {
                        if (l[i] > max_l) {
                            l[max_i] = -INFINITY;
                            max_i = i;
                            max_l = l[i];
                        } else {
                            l[i] = -INFINITY;
                        }
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        l[i] = GGML_DRAFT_PICK_DIVF(l[i], t);
                    }
                }
            }
        }

        if (n > 0) {
            // LLAMA_SPEC_REJECTION_TEMP
            const float tr = prm[28];
            if (tr > 0.0f && prm[4] > 0.0f && tr != prm[4]) {
                const float s = GGML_DRAFT_PICK_DIVF(prm[4], tr);
                for (int i = 0; i < n; ++i) {
                    if (ggml_draft_pick_isfinite(l[i])) {
                        l[i] *= s;
                    }
                }
            }
            float max_l = -INFINITY;
            for (int i = 0; i < n; ++i) {
                max_l = l[i] > max_l ? l[i] : max_l;
            }
            if (ggml_draft_pick_isfinite(max_l)) {
                double sum = 0.0;
                for (int i = 0; i < n; ++i) {
                    p[i] = ggml_draft_pick_isfinite(l[i]) ? GGML_DRAFT_PICK_EXPF(l[i] - max_l) : 0.0f;
                    sum += p[i];
                }
                for (int i = 0; i < n; ++i) {
                    p[i] = (float) (p[i] / sum);
                }
                // the inverse CDF of u over the weights p, in the array's order
                double tot = 0.0;
                int last = -1;
                for (int i = 0; i < n; ++i) {
                    if ((double) p[i] > 0.0) {
                        tot += (double) p[i];
                        last = i;
                    }
                }
                if (last >= 0) {
                    const double tgt = ((double) prm[6] + (double) prm[29]) * tot;
                    double run = 0.0;
                    sel = last;
                    for (int i = 0; i < n; ++i) {
                        if ((double) p[i] > 0.0) {
                            run += (double) p[i];
                            if (run >= tgt) {
                                sel = i;
                                break;
                            }
                        }
                    }
                    int nq = 0;
                    for (int i = 0; i < n; ++i) {
                        if (p[i] > 0.0f) {
                            out[GGML_DRAFT_PICK_HDR + nq]     = ids[e[i]];
                            out[GGML_DRAFT_PICK_HDR + k + nq] = ggml_draft_pick_f2i(p[i]);
                            nq++;
                        }
                    }
                    out[0] = ids[e[sel]];
                    out[1] = rows[e[sel]];
                    out[2] = ggml_draft_pick_f2i(p[sel]);
                    out[4] = nq;
                    out[5] = e[sel];
                    return;
                }
            }
        }
        none = true;
    }

    // the argmax: the first candidate of the draft's own order, with p0 (no q row)
    if (none) {
        // the arrays may have been reordered by the chain: find the draft order's first again (the highest raw logit,
        // ties earlier first)
        int best = 0;
        for (int i = 1; i < k; ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        out[0] = ids[best];
        out[1] = rows[best];
        out[2] = out[3];
        out[4] = 0;
        out[5] = best;
        out[6] = mode == 1 ? 1 : 0;
    }
}
