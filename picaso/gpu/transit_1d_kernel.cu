#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        exit(1);                                                             \
    }                                                                        \
} while (0)

__device__ __forceinline__ double safe_sqrt(double x) {
    return sqrt(fmax(x, 0.0));
}

extern "C" __global__ void k_build_delta_length(
    const double* __restrict__ z,            // (nlevel)
    const double* __restrict__ player,       // (nlayer)
    const double* __restrict__ tlayer,       // (nlayer)
    double k_b,
    int nlevel,
    double* __restrict__ delta_length        // (nlevel, nlevel)
){
    int i = (int)blockIdx.y;
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= nlevel || j >= nlevel) return;

    if (j >= i) {
        delta_length[(int64_t)i * (int64_t)nlevel + (int64_t)j] = 0.0;
        return;
    }

    int idx = i - j - 1;  // layer index: 0..nlevel-2
    double T = tlayer[idx];

    if (k_b <= 0.0 || T <= 0.0) {
        delta_length[(int64_t)i * (int64_t)nlevel + (int64_t)j] = 0.0;
        return;
    }

    double reference = z[i];
    double inner     = z[i - j];
    double outer     = z[i - j - 1];

    double a = outer * outer - reference * reference;
    double b = inner * inner - reference * reference;

    double seg  = safe_sqrt(a) - safe_sqrt(b);
    double dens = player[idx] / (T * k_b);

    delta_length[(int64_t)i * (int64_t)nlevel + (int64_t)j] = seg * dens;
}

extern "C" __global__ void k_compute_mmw_g(
    const double* __restrict__ mmw,
    double amu,
    int nlayer,
    double* __restrict__ mmw_g
){
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (k < nlayer) {
        mmw_g[k] = mmw[k] * amu;
    }
}

extern "C" __global__ void k_compute_transmitted(
    const double* __restrict__ DTAU,          // (nlayer, nwno) row-major
    const double* __restrict__ colden,        // (nlayer)
    const double* __restrict__ mmw_g,         // (nlayer)
    const double* __restrict__ delta_length,  // (nlevel, nlevel)
    int nlevel,
    int nlayer,
    int nwno,
    double* __restrict__ transmitted          // (nwno, nlevel)
){
    int w = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)blockIdx.y;

    if (w >= nwno || i >= nlevel) return;

    double tau = 0.0;

    for (int j = 0; j < i; ++j) {
        int k = i - j - 1;

        double cd = colden[k];
        if (cd <= 0.0) continue;

        double taucoeff =
            (DTAU[(int64_t)k * (int64_t)nwno + (int64_t)w] / cd) * mmw_g[k];

        double dl =
            delta_length[(int64_t)i * (int64_t)nlevel + (int64_t)j];

        tau += 2.0 * taucoeff * dl;
    }

    transmitted[(int64_t)w * (int64_t)nlevel + (int64_t)i] = exp(-tau);
}

extern "C" __global__ void k_reduce_F(
    const double* __restrict__ transmitted,   // (nwno, nlevel)
    const double* __restrict__ z,             // (nlevel)
    const double* __restrict__ dz,            // (nlevel)
    int nlevel,
    int nwno,
    double zmin,
    double rstar,
    double* __restrict__ F                    // (nwno)
){
    int w = (int)blockIdx.x;
    if (w >= nwno) return;

    double sum = 0.0;

    for (int i = (int)threadIdx.x; i < nlevel; i += (int)blockDim.x) {
        double T = transmitted[(int64_t)w * (int64_t)nlevel + (int64_t)i];
        sum += (1.0 - T) * z[i] * dz[i];
    }

    __shared__ double sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double rs2 = rstar * rstar;
        double zmr = zmin / rstar;
        F[w] = zmr * zmr + (2.0 / rs2) * sh[0];
    }
}


struct TransitContext {
    bool initialized = false;

    int nlevel = 0;
    int nlayer = 0;
    int nwno   = 0;

    double k_b   = 0.0;
    double amu   = 0.0;
    double rstar = 0.0;
    double zmin  = 0.0;

    const double* z_dev      = nullptr;
    const double* dz_dev     = nullptr;
    const double* player_dev = nullptr;
    const double* tlayer_dev = nullptr;
    const double* colden_dev = nullptr;
    const double* DTAU_dev   = nullptr;
    const double* mmw_dev    = nullptr;

    double* mmw_g_dev        = nullptr;
    double* delta_length_dev = nullptr;
    double* transmitted_dev  = nullptr;
};

static TransitContext g_ctx;

extern "C" void get_transit_1d_free();

extern "C" void get_transit_1d_allocate_buffers(
    int nlevel,
    int nwno
){
    TransitContext& ctx = g_ctx;

    if (ctx.initialized) {
        get_transit_1d_free();
    }

    ctx.nlevel = nlevel;
    ctx.nlayer = nlevel - 1;
    ctx.nwno   = nwno;

    size_t layer_bytes = (size_t)ctx.nlayer * sizeof(double);
    size_t delta_bytes = (size_t)nlevel * (size_t)nlevel * sizeof(double);
    size_t trans_bytes = (size_t)nwno * (size_t)nlevel * sizeof(double);

    CUDA_CHECK(cudaMalloc(&ctx.mmw_g_dev,        layer_bytes));
    CUDA_CHECK(cudaMalloc(&ctx.delta_length_dev, delta_bytes));
    CUDA_CHECK(cudaMalloc(&ctx.transmitted_dev,  trans_bytes));

    ctx.initialized = true;
}

extern "C" void get_transit_1d_set_inputs(
    const double* z,
    const double* dz,
    const double* player,
    const double* tlayer,
    const double* colden,
    const double* DTAU,
    const double* mmw,
    double k_b,
    double amu,
    double rstar,
    int nlevel,
    int nwno
){
    TransitContext& ctx = g_ctx;

    ctx.nlevel = nlevel;
    ctx.nlayer = nlevel - 1;
    ctx.nwno   = nwno;

    ctx.z_dev      = z;
    ctx.dz_dev     = dz;
    ctx.player_dev = player;
    ctx.tlayer_dev = tlayer;
    ctx.colden_dev = colden;
    ctx.DTAU_dev   = DTAU;
    ctx.mmw_dev    = mmw;

    ctx.k_b   = k_b;
    ctx.amu   = amu;
    ctx.rstar = rstar;

    double* z_h = (double*)malloc((size_t)nlevel * sizeof(double));
    if (!z_h) {
        fprintf(stderr, "Failed to allocate temporary host memory for zmin.\n");
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(
        z_h, z, (size_t)nlevel * sizeof(double), cudaMemcpyDeviceToHost));

    ctx.zmin = z_h[0];
    for (int i = 1; i < nlevel; ++i) {
        if (z_h[i] < ctx.zmin) ctx.zmin = z_h[i];
    }
    free(z_h);

    int block_layer = 256;
    int grid_layer  = (ctx.nlayer + block_layer - 1) / block_layer;

    k_compute_mmw_g<<<grid_layer, block_layer>>>(
        ctx.mmw_dev, ctx.amu, ctx.nlayer, ctx.mmw_g_dev);
    CUDA_CHECK(cudaGetLastError());

    int TX = 128;
    dim3 blockDL(TX, 1, 1);
    dim3 gridDL((ctx.nlevel + TX - 1) / TX, ctx.nlevel, 1);

    k_build_delta_length<<<gridDL, blockDL>>>(
        ctx.z_dev,
        ctx.player_dev,
        ctx.tlayer_dev,
        ctx.k_b,
        ctx.nlevel,
        ctx.delta_length_dev
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void get_transit_1d_run(
    double* F_out_dev
){
    TransitContext& ctx = g_ctx;

    if (!ctx.initialized) {
        fprintf(stderr, "get_transit_1d_run called before init\n");
        return;
    }

    dim3 block1(128, 1, 1);
    dim3 grid1((ctx.nwno + 127) / 128, ctx.nlevel, 1);

    dim3 block2(256, 1, 1);
    dim3 grid2(ctx.nwno, 1, 1);

    k_compute_transmitted<<<grid1, block1>>>(
        ctx.DTAU_dev,
        ctx.colden_dev,
        ctx.mmw_g_dev,
        ctx.delta_length_dev,
        ctx.nlevel,
        ctx.nlayer,
        ctx.nwno,
        ctx.transmitted_dev
    );
    CUDA_CHECK(cudaGetLastError());

    k_reduce_F<<<grid2, block2>>>(
        ctx.transmitted_dev,
        ctx.z_dev,
        ctx.dz_dev,
        ctx.nlevel,
        ctx.nwno,
        ctx.zmin,
        ctx.rstar,
        F_out_dev
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void get_transit_1d_free()
{
    TransitContext& ctx = g_ctx;
    if (!ctx.initialized) return;

    auto FREE = [](double*& p) {
        if (p) {
            CUDA_CHECK(cudaFree(p));
            p = nullptr;
        }
    };

    FREE(ctx.mmw_g_dev);
    FREE(ctx.delta_length_dev);
    FREE(ctx.transmitted_dev);

    ctx.z_dev      = nullptr;
    ctx.dz_dev     = nullptr;
    ctx.player_dev = nullptr;
    ctx.tlayer_dev = nullptr;
    ctx.colden_dev = nullptr;
    ctx.DTAU_dev   = nullptr;
    ctx.mmw_dev    = nullptr;

    ctx.initialized = false;
}
