// CUDA kernels for the PyTorches CUDA plugin. Compiled to PTX by scripts/build-cuda-kernels.ps1;
// the generated kernels.ptx is committed and embedded in the plugin (JIT-compiled by the driver).
//
// All indexing is 64-bit. Elementwise kernels use grid-stride loops.

typedef unsigned long long u64;
typedef long long i64;

// Strided operand description; layout must match `Dims` in src/lib.rs.
struct Dims {
    unsigned int nd;
    unsigned int contig;  // 1 if offset == linear index
    u64 shape[8];
    u64 strides[8];
};

__device__ __forceinline__ u64 offset_of(const Dims& d, u64 i) {
    if (d.contig) return i;
    u64 off = 0;
    for (int k = (int)d.nd - 1; k >= 0; --k) {
        u64 s = d.shape[k];
        u64 q = i / s;
        off += (i - q * s) * d.strides[k];
        i = q;
    }
    return off;
}

// Op codes (subset used here; the Rust side maps ABI op codes to these).
enum { U_NEG = 1, U_EXP, U_LOG, U_RELU, U_TANH, U_STEP, U_COPY };
enum { B_ADD = 0, B_SUB, B_MUL, B_DIV };

extern "C" __global__ void unary_k(int op, const float* __restrict__ in, float* __restrict__ out, u64 n, Dims d) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = in[offset_of(d, i)];
        float r;
        switch (op) {
            case U_NEG: r = -v; break;
            case U_EXP: r = expf(v); break;
            case U_LOG: r = logf(v); break;
            case U_RELU: r = fmaxf(v, 0.0f); break;
            case U_TANH: r = tanhf(v); break;
            case U_STEP: r = v > 0.0f ? 1.0f : 0.0f; break;
            default: r = v; break;
        }
        out[i] = r;
    }
}

extern "C" __global__ void binary_k(int op, const float* __restrict__ a, const float* __restrict__ b,
                                    float* __restrict__ out, u64 n, Dims da, Dims db) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float x = a[offset_of(da, i)];
        float y = b[offset_of(db, i)];
        float r;
        switch (op) {
            case B_ADD: r = x + y; break;
            case B_SUB: r = x - y; break;
            case B_MUL: r = x * y; break;
            default: r = x / y; break;
        }
        out[i] = r;
    }
}

// Contiguous same-shape binary op over 128-bit vectors. Launched only when n % 4 == 0 and all three
// pointers are 16-byte aligned; everything else goes through binary_k.
extern "C" __global__ void binary_vec_k(int op, const float4* __restrict__ a, const float4* __restrict__ b,
                                        float4* __restrict__ out, u64 n4) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 x = a[i], y = b[i], r;
        switch (op) {
            case B_ADD: r = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w); break;
            case B_SUB: r = make_float4(x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w); break;
            case B_MUL: r = make_float4(x.x * y.x, x.y * y.y, x.z * y.z, x.w * y.w); break;
            default: r = make_float4(x.x / y.x, x.y / y.y, x.z / y.z, x.w / y.w); break;
        }
        out[i] = r;
    }
}

// out = a + alpha * b. `out` may alias `a` (in-place update), so no __restrict__ on those two.
extern "C" __global__ void axpy_k(const float* a, const float* __restrict__ b, float* out, u64 n, float alpha) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = a[i] + alpha * b[i];
}

extern "C" __global__ void fill_k(float* __restrict__ out, u64 n, float v) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = v;
}

__device__ __forceinline__ u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void randn_k(float* __restrict__ out, u64 n, u64 seed) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        u64 h = splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ULL));
        float u1 = (float)((h >> 40) + 1) / 16777216.0f;
        float u2 = (float)(h & 0xFFFFFFULL) / 16777216.0f;
        out[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    }
}

// ---- matmul: C[m,n] = A[m,k] * B[k,n], row-major, contiguous ------------------------------
// 128x128 block tile, BK=8, 256 threads, 8x8 register micro-tile per thread.
#define BM 128
#define BN 128
#define BK 8
#define APAD (BM + 4)

extern "C" __global__ void __launch_bounds__(256) matmul_k(const float* __restrict__ A, const float* __restrict__ B,
                                                           float* __restrict__ C, u64 M, u64 N, u64 K) {
    __shared__ float As[BK][APAD];  // transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const u64 row0 = (u64)blockIdx.y * BM;
    const u64 col0 = (u64)blockIdx.x * BN;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    for (u64 k0 = 0; k0 < K; k0 += BK) {
        // A tile: 128 x 8, 4 elements per thread.
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int idx = tid + 256 * l;
            int m = idx >> 3, kk = idx & 7;
            u64 gr = row0 + m, gk = k0 + kk;
            As[kk][m] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }
        // B tile: 8 x 128, 4 elements per thread (coalesced along n).
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int idx = tid + 256 * l;
            int kk = idx >> 7, n = idx & 127;
            u64 gk = k0 + kk, gc = col0 + n;
            Bs[kk][n] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[8], b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[kk][ty * 8 + i];
#pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        u64 gr = row0 + ty * 8 + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            u64 gc = col0 + tx + 16 * j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// ---- reductions ----------------------------------------------------------------------------

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();  // protect warp_sums reuse
    if (lane == 0) warp_sums[w] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? warp_sums[threadIdx.x] : 0.0f;
    if (w == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    return v;  // valid in thread 0
}

// inner == 1: row o of x[outer][n]; blockIdx.x = row, blockIdx.y = chunk of the row.
// Writes dst[chunk * outer + row] (dst is `out` directly when gridDim.y == 1).
extern "C" __global__ void sum_rows_k(const float* __restrict__ x, float* __restrict__ dst, u64 outer, u64 n, u64 chunk,
                                       int vec) {
    u64 row = blockIdx.x;
    u64 begin = (u64)blockIdx.y * chunk;
    u64 end = begin + chunk < n ? begin + chunk : n;
    const float* p = x + row * n;
    float s = 0.0f;
    if (vec) {
        // Host guarantees n % 4 == 0, chunk % 4 == 0 and a 16-byte aligned x, so every row and chunk
        // start is aligned and `begin`/`end` are multiples of 4 (end == n when it is clamped).
        const float4* p4 = reinterpret_cast<const float4*>(p);
        for (u64 j = begin / 4 + threadIdx.x; j < end / 4; j += blockDim.x) {
            float4 v = p4[j];
            s += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (u64 j = begin + threadIdx.x; j < end; j += blockDim.x) s += p[j];
    }
    s = block_reduce_sum(s);
    if (threadIdx.x == 0) dst[(u64)blockIdx.y * outer + row] = s;
}

// inner > 1: one thread per output element (o, i); blockIdx.y = chunk of the n range.
// Writes dst[chunk * (outer*inner) + o*inner + i].
extern "C" __global__ void sum_cols_k(const float* __restrict__ x, float* __restrict__ dst, u64 outer, u64 n, u64 inner,
                                      u64 chunk) {
    u64 total = outer * inner;
    u64 stride = (u64)gridDim.x * blockDim.x;
    u64 begin = (u64)blockIdx.y * chunk;
    u64 end = begin + chunk < n ? begin + chunk : n;
    for (u64 id = (u64)blockIdx.x * blockDim.x + threadIdx.x; id < total; id += stride) {
        u64 o = id / inner, i = id - o * inner;
        const float* p = x + (o * n + begin) * inner + i;
        float s = 0.0f;
        for (u64 j = begin; j < end; ++j, p += inner) s += *p;
        dst[(u64)blockIdx.y * total + id] = s;
    }
}

// Second pass: out[id] = sum over s < parts of part[s * total + id].
extern "C" __global__ void sum_parts_k(const float* __restrict__ part, float* __restrict__ out, u64 total, u64 parts) {
    u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 id = (u64)blockIdx.x * blockDim.x + threadIdx.x; id < total; id += stride) {
        float s = 0.0f;
        for (u64 p = 0; p < parts; ++p) s += part[p * total + id];
        out[id] = s;
    }
}
