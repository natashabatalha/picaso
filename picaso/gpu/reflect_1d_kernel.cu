#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(err));                \
        exit(1);                                                             \
    }                                                                        \
} while (0)

using namespace std;

// -----------------------------------------------------------------------------
// Common helpers
// -----------------------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__device__ __forceinline__ int global_idx_1d() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}


static inline dim3 make_grid(int N) {
    return dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
}


// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

__global__ void calculate_g12_lg_eddi(
    double * __restrict__ w0_dev,
    double * __restrict__ ftau_cld_dev,
    double * __restrict__ cosb_dev,
    int vec_size,
    double * __restrict__ g1_dev,
    double * __restrict__ g2_dev,
    double * __restrict__ lambda_dev,
    double * __restrict__ gama_dev)
{
    int index = global_idx_1d();
    if (index >= vec_size) return;

    const double w0   = w0_dev[index];
    const double ftau = ftau_cld_dev[index];
    const double cosb = cosb_dev[index];

    const double g1 = (7.0 - w0 * (4.0 + 3.0 * ftau * cosb)) / 4.0;
    const double g2 = -(1.0 - w0 * (4.0 - 3.0 * ftau * cosb)) / 4.0;

    const double g1_sq = g1 * g1;
    const double g2_sq = g2 * g2;
    const double lam   = sqrt(fmax(g1_sq - g2_sq, 0.0));

    g1_dev[index]     = g1;
    g2_dev[index]     = g2;
    lambda_dev[index] = lam;
    gama_dev[index]   = (g1 - lam) / g2;
}


__global__ void calculate_g12_lg_quad(
    double * __restrict__ w0_dev,
    double * __restrict__ ftau_cld_dev,
    double * __restrict__ cosb_dev,
    int vec_size,
    double * __restrict__ g1_dev,
    double * __restrict__ g2_dev,
    double * __restrict__ lambda_dev,
    double * __restrict__ gama_dev)
{
    int index = global_idx_1d();
    if (index >= vec_size) return;

    const double sq3  = 1.7320508075688772; // sqrt(3)
    const double w0   = w0_dev[index];
    const double ftau = ftau_cld_dev[index];
    const double cosb = cosb_dev[index];

    const double tmp = ftau * cosb;

    const double g1 = 0.5 * sq3 * (2.0 - w0 * (1.0 + tmp));
    const double g2 = 0.5 * sq3 * w0 * (1.0 - tmp);

    const double g1_sq = g1 * g1;
    const double g2_sq = g2 * g2;
    const double lam   = sqrt(fmax(g1_sq - g2_sq, 0.0));

    g1_dev[index]     = g1;
    g2_dev[index]     = g2;
    lambda_dev[index] = lam;
    gama_dev[index]   = (g1 - lam) / g2;
}


__global__ void calculate_all(
    double * __restrict__ w0_dev,
    double * __restrict__ ftau_cld_dev,
    double * __restrict__ cosb_dev,
    int vec_size,
    double * __restrict__ g1_dev,
    double * __restrict__ g2_dev,
    double * __restrict__ lambda_dev,
    double * __restrict__ g3_dev,
    double * __restrict__ g4_dev,
    double * __restrict__ c_plus_up_dev,
    double * __restrict__ c_plus_down_dev,
    double * __restrict__ c_minus_up_dev,
    double * __restrict__ c_minus_down_dev,
    double * __restrict__ b_surface_dev,
    double u0,
    double * __restrict__ tau_dev,
    int wv_len,
    double * __restrict__ f0pi_dev,
    double * __restrict__ atm_surf_reflect_dev,
    double * __restrict__ dtau_dev,
    double * __restrict__ exptrm_dev,
    double * __restrict__ a_minus_dev,
    double * __restrict__ a_plus_dev)
{
    int index = global_idx_1d();
    int index_1d = (wv_len > 0) ? (index % wv_len) : 0;

    const double sq3 = 1.7320508075688772;
    const double u0_inv = 1.0 / u0;

    if (index < vec_size) {
        const double ftau = ftau_cld_dev[index];
        const double cosb = cosb_dev[index];

        const double g3 = 0.5 * (1.0 - sq3 * ftau * cosb * u0);
        const double g4 = 1.0 - g3;

        g3_dev[index] = g3;
        g4_dev[index] = g4;

        const double g1_val = g1_dev[index];
        const double g2_val = g2_dev[index];
        const double lambda = lambda_dev[index];

        const double denom = (lambda * lambda) - (u0_inv * u0_inv);
        const double w0    = w0_dev[index];
        const double f0    = f0pi_dev[index_1d];

        const double tmp1 = g4 * (g1_val + u0_inv) + g2_val * g3;
        const double tmp2 = g3 * (g1_val - u0_inv) + g2_val * g4;
        const double pref = f0 * w0;

        const double a_minus = pref * tmp1 / denom;
        const double a_plus  = pref * tmp2 / denom;

        const double tau_here     = tau_dev[index];
        const double tau_next_lvl = tau_dev[index + wv_len];

        const double exp_top = exp(-tau_here * u0_inv);
        const double exp_bot = exp(-tau_next_lvl * u0_inv);

        c_minus_up_dev[index]   = a_minus * exp_top;
        c_plus_up_dev[index]    = a_plus  * exp_top;
        c_minus_down_dev[index] = a_minus * exp_bot;
        c_plus_down_dev[index]  = a_plus  * exp_bot;

        exptrm_dev[index] = lambda * dtau_dev[index];
        a_minus_dev[index] = a_minus;
        a_plus_dev[index]  = a_plus;
    }

    if (index < wv_len) {
        // surface term
        b_surface_dev[index] = 0.0 + atm_surf_reflect_dev[index] * u0 * f0pi_dev[index] *
                               exp(-tau_dev[vec_size + index] * u0_inv);
    }
}


__global__ void calculate_e_matrix(
    double * __restrict__ exptrm_positive_dev,
    double * __restrict__ exptrm_minus_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ e1_dev,
    double * __restrict__ e2_dev,
    double * __restrict__ e3_dev,
    double * __restrict__ e4_dev,
    int vec_size)
{
    int index = global_idx_1d();
    if (index >= vec_size) return;

    const double ep = exptrm_positive_dev[index];
    const double em = exptrm_minus_dev[index];
    const double g  = gama_dev[index];

    e1_dev[index] = ep + g * em;
    e2_dev[index] = ep - g * em;
    e3_dev[index] = g * ep + em;
    e4_dev[index] = g * ep - em;
}


__global__ void setup_tri_diag_1st(
    int nlayer, int nwno,
    double * __restrict__ c_plus_up_dev,
    double * __restrict__ c_minus_up_dev,
    double * __restrict__ c_plus_down_dev,
    double * __restrict__ c_minus_down_dev,
    double b_top,
    double * __restrict__ b_surface_dev,
    double * __restrict__ atm_surf_reflect_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ dtau_dev,
    double * __restrict__ e1_dev,
    double * __restrict__ e2_dev,
    double * __restrict__ e3_dev,
    double * __restrict__ e4_dev,
    double * __restrict__ A_odd_dev,
    double * __restrict__ B_odd_dev,
    double * __restrict__ C_odd_dev,
    double * __restrict__ D_odd_dev)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    A_odd_dev[index] = 0.0;
    B_odd_dev[index] = gama_dev[index] + 1.0;
    C_odd_dev[index] = gama_dev[index] - 1.0;
    D_odd_dev[index] = b_top - c_minus_up_dev[index];
}


__global__ void setup_tri_diag_last(
    int nlayer, int nwno,
    double * __restrict__ c_plus_up_dev,
    double * __restrict__ c_minus_up_dev,
    double * __restrict__ c_plus_down_dev,
    double * __restrict__ c_minus_down_dev,
    double b_top,
    double * __restrict__ b_surface_dev,
    double * __restrict__ atm_surf_reflect_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ dtau_dev,
    double * __restrict__ e1_dev,
    double * __restrict__ e2_dev,
    double * __restrict__ e3_dev,
    double * __restrict__ e4_dev,
    double * __restrict__ A_odd_dev,
    double * __restrict__ B_odd_dev,
    double * __restrict__ C_odd_dev,
    double * __restrict__ D_odd_dev)
{
    int index = global_idx_1d();
    int total = nlayer * nwno;
    if (index >= total) return;

    if ((index > (nlayer - 1) * nwno) && (index < nlayer * nwno)) {
        int idx = index + nlayer * nwno;
        A_odd_dev[idx] = e1_dev[index] - atm_surf_reflect_dev[index] * e3_dev[index];
        B_odd_dev[idx] = e2_dev[index] - atm_surf_reflect_dev[index] * e4_dev[index];
        C_odd_dev[idx] = 0.0;
        D_odd_dev[idx] = b_surface_dev[index - (nlayer - 1) * nwno]
                         - c_plus_down_dev[index]
                         + atm_surf_reflect_dev[index] * c_minus_down_dev[index];
    }
}


__global__ void init_matrix(
    double * __restrict__ AS_dev,
    double * __restrict__ DS_dev,
    double * __restrict__ A_odd_dev,
    double * __restrict__ B_odd_dev,
    double * __restrict__ D_odd_dev,
    int nlayer, int nwno)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    int index_last = index + (2 * nlayer - 1) * nwno;
    const double denom = B_odd_dev[index_last] + 1e-16;
    AS_dev[index_last] = A_odd_dev[index_last] / denom;
    DS_dev[index_last] = D_odd_dev[index_last] / denom;
}


__global__ void calculate_Xmatrix1(
    double * __restrict__ AS_dev,
    double * __restrict__ DS_dev,
    double * __restrict__ XK_dev,
    double * __restrict__ A_odd_dev,
    double * __restrict__ B_odd_dev,
    double * __restrict__ C_odd_dev,
    double * __restrict__ D_odd_dev,
    int nlayer, int nwno,
    int i_layer)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    int index1 = index + i_layer * nwno;
    int index2 = index + (i_layer + 1) * nwno;

    const double x_val = 1.0 / (B_odd_dev[index1] - C_odd_dev[index1] * AS_dev[index2]);
    AS_dev[index1] = A_odd_dev[index1] * x_val;
    DS_dev[index1] = (D_odd_dev[index1] - C_odd_dev[index1] * DS_dev[index2]) * x_val;
}


__global__ void calculate_Xmatrix2(
    double * __restrict__ AS_dev,
    double * __restrict__ DS_dev,
    double * __restrict__ XK_dev,
    int nlayer, int nwno,
    int i_layer)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    int index1 = index + i_layer * nwno;
    int index2 = index + (i_layer - 1) * nwno;

    XK_dev[index1] = DS_dev[index1] - AS_dev[index1] * XK_dev[index2];
}


__global__ void set_matrix_zero(
    double * __restrict__ XK_dev,
    double * __restrict__ DS_dev,
    int nlayer, int nwno)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    XK_dev[index] = DS_dev[index];
}


__global__ void calculate_pos_neg(
    double * __restrict__ XK_dev,
    double * __restrict__ positive_dev,
    double * __restrict__ negative_dev,
    int nlayer, int nwno)
{
    int index = global_idx_1d();
    if (index >= nlayer * nwno) return;

    int index_wn    = index % nwno;
    int index_layer = index / nwno;

    int index_odd_sav  = nwno * (index_layer * 2 + 1) + index_wn;
    int index_even_sav = nwno * (index_layer * 2 + 0) + index_wn;

    const double XK_even = XK_dev[index_even_sav];
    const double XK_odd  = XK_dev[index_odd_sav];

    positive_dev[index] = XK_even + XK_odd;
    negative_dev[index] = XK_even - XK_odd;
}


__global__ void setup_tri_diag_all(
    int nlayer, int nwno,
    double * __restrict__ c_plus_up_dev,
    double * __restrict__ c_minus_up_dev,
    double * __restrict__ c_plus_down_dev,
    double * __restrict__ c_minus_down_dev,
    double b_top,
    double * __restrict__ b_surface_dev,
    double * __restrict__ atm_surf_reflect_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ dtau_dev,
    double * __restrict__ e1_dev,
    double * __restrict__ e2_dev,
    double * __restrict__ e3_dev,
    double * __restrict__ e4_dev,
    double * __restrict__ A_odd_dev,
    double * __restrict__ B_odd_dev,
    double * __restrict__ C_odd_dev,
    double * __restrict__ D_odd_dev)
{
    int index = global_idx_1d();
    int total = nlayer * nwno;
    if (index >= total) return;

    int index_wn    = index % nwno;
    int index_layer = index / nwno;

    if (index + nwno < total) {
        int index_odd_sav  = nwno * (index_layer * 2 + 1) + index_wn;
        int index_even_sav = nwno * (index_layer * 2 + 2) + index_wn;

        const double g_next = gama_dev[index + nwno];
        const double g_here = gama_dev[index];

        const double g_next_sq = g_next * g_next;
        const double g_here_sq = g_here * g_here;

        const double e1 = e1_dev[index];
        const double e2 = e2_dev[index];
        const double e3 = e3_dev[index];
        const double e4 = e4_dev[index];

        const double cp_up_next   = c_plus_up_dev[index + nwno];
        const double cm_up_next   = c_minus_up_dev[index + nwno];
        const double cp_down_here = c_plus_down_dev[index];
        const double cm_down_here = c_minus_down_dev[index];

        // odd rows
        A_odd_dev[index_odd_sav] = (e1 + e3) * (g_next - 1.0);
        B_odd_dev[index_odd_sav] = (e2 + e4) * (g_next - 1.0);
        C_odd_dev[index_odd_sav] = 2.0 * (1.0 - g_next_sq);
        D_odd_dev[index_odd_sav] =
            ((g_next - 1.0) * (cp_up_next - cp_down_here)
             + (1.0 - g_next) * (cm_down_here - cm_up_next));

        // even rows
        A_odd_dev[index_even_sav] = 2.0 * (1.0 - g_here_sq);
        B_odd_dev[index_even_sav] = (e1 - e3) * (g_next + 1.0);
        C_odd_dev[index_even_sav] = (e1 + e3) * (g_next - 1.0);
        D_odd_dev[index_even_sav] =
            (e3 * (cp_up_next - cp_down_here) +
             e1 * (cm_down_here - cm_up_next));
    }
}


__global__ void calculate_allexptrm(
    double * __restrict__ exptrm_dev,
    double * __restrict__ exptrm_positive_dev,
    double * __restrict__ exptrm_minus_dev,
    int cutt_off,
    int vec_size)
{
    int index = global_idx_1d();
    if (index >= vec_size) return;

    double x = exptrm_dev[index];
    double ep;

    if (x >= 35.0) {
        ep = exp(35.0);
        exptrm_dev[index] = 35.0;
    } else {
        ep = exp(x);
    }
    double em = 1.0 / ep;

    exptrm_positive_dev[index] = ep;
    exptrm_minus_dev[index]    = em;
}


__global__ void calculate_flux0(
    double * __restrict__ xint_dev,
    double * __restrict__ positive_dev,
    double * __restrict__ negative_dev,
    double * __restrict__ exptrm_positive_dev,
    double * __restrict__ exptrm_minus_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ c_plus_down_dev,
    int nlayer, int nwno)
{
    const double pi = 3.141592653589793238463;
    int index = global_idx_1d();

    int total = nlayer * nwno;
    if (index < total && index >= (nlayer - 1) * nwno) {
        int idx = index;
        xint_dev[index + nwno] =
            (positive_dev[idx] * exptrm_positive_dev[idx]
             + gama_dev[idx] * negative_dev[idx] * exptrm_minus_dev[idx]
             + c_plus_down_dev[idx]) / pi;
    }
}


__global__ void calculate_GHA(
    double * __restrict__ G_matrix_dev,
    double * __restrict__ H_matrix_dev,
    double * __restrict__ A_matrix_dev,
    double * __restrict__ ftau_cld_dev,
    double * __restrict__ cosb_dev,
    double * __restrict__ gcos2_dev,
    double * __restrict__ positive_dev,
    double * __restrict__ negative_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ w0_dev,
    double * __restrict__ c_plus_up_dev,
    double * __restrict__ c_minus_up_dev,
    double ubar1,
    int nlayer, int nwno)
{
    const double pi   = 3.141592653589793238463;
    const double ubar2 = 0.767;

    int index = global_idx_1d();
    int total = nlayer * nwno;
    if (index >= total) return;

    const double ftau  = ftau_cld_dev[index];
    const double cosb  = cosb_dev[index];
    const double gcos2 = gcos2_dev[index];

    const double pos = positive_dev[index];
    const double neg = negative_dev[index];
    const double g   = gama_dev[index];
    const double w0  = w0_dev[index];

    // 3 * u2^2 * u1^2 - 1
    const double u1_sq = ubar1 * ubar1;
    const double u2_sq = ubar2 * ubar2;
    const double term_leg = (3.0 * u2_sq * u1_sq - 1.0) / 2.0;

    const double multi_plus  = (1.0 + 1.5 * ftau * cosb * ubar1 + gcos2 * term_leg);
    const double multi_minus = (1.0 - 1.5 * ftau * cosb * ubar1 + gcos2 * term_leg);

    G_matrix_dev[index] = pos * (multi_plus + g * multi_minus)   * w0 * 0.5 / pi;
    H_matrix_dev[index] = neg * (g * multi_plus + multi_minus)   * w0 * 0.5 / pi;
    A_matrix_dev[index] =
        (multi_plus * c_plus_up_dev[index] +
         multi_minus * c_minus_up_dev[index]) * w0 * 0.5 / pi;
}


__global__ void direct_scattering(
    double constant_forward,
    double constant_back,
    double frac_a, double frac_b, double frac_c,
    double cos_theta,
    double * __restrict__ cosb_og_dev,
    double * __restrict__ ftau_cld_dev,
    double * __restrict__ ftau_ray_dev,
    double * __restrict__ p_single_dev,
    int nlayer, int nwno)
{
    int index = global_idx_1d();
    int total = nlayer * nwno;
    if (index >= total) return;

    const double cosb_og = cosb_og_dev[index];
    const double ftau_c  = ftau_cld_dev[index];
    const double ftau_r  = ftau_ray_dev[index];

    const double g_forward = constant_forward * cosb_og;
    const double g_back    = constant_back    * cosb_og;

    const double f = frac_a + frac_b * pow(g_back, frac_c); // exponent is not constant

    const double cos2 = cos_theta * cos_theta;

    // HG_forward
    double tmp_f = 1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta;
    double denom_f = sqrt(tmp_f * tmp_f * tmp_f);
    double HG_forward = (1.0 - g_forward * g_forward) / denom_f;

    // HG_back
    double tmp_b = 1.0 + g_back * g_back + 2.0 * g_back * cos_theta;
    double denom_b = sqrt(tmp_b * tmp_b * tmp_b);
    double HG_back = (1.0 - g_back * g_back) / denom_b;

    double ray_phase = 0.75 * (1.0 + cos2);

    p_single_dev[index] =
        ftau_c * (f * HG_forward + (1.0 - f) * HG_back)
        + ftau_r * ray_phase;
}


__global__ void calculate_xint(
    double * __restrict__ xint_dev,
    double * __restrict__ dtau_og_dev,
    double * __restrict__ dtau_dev,
    double * __restrict__ w0_og_dev,
    double * __restrict__ f0pi_dev,
    double * __restrict__ p_single_dev,
    double * __restrict__ tau_og_dev,
    double * __restrict__ A_matrix_dev,
    double * __restrict__ G_matrix_dev,
    double * __restrict__ H_matrix_dev,
    double * __restrict__ exptrm_dev,
    double * __restrict__ lambda_dev,
    double ubar0, double ubar1,
    int i_layer, int nwno)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    const double pi = 3.141592653589793238463;

    int index_sav  = index + i_layer       * nwno;
    int index_read = index + (i_layer + 1) * nwno;

    const double dtau_og = dtau_og_dev[index_sav];
    const double dtau    = dtau_dev[index_sav];
    const double w0_og   = w0_og_dev[index_sav];
    const double f0      = f0pi_dev[index];
    const double p_single = p_single_dev[index_sav];
    const double tau_og   = tau_og_dev[index_sav];
    const double A        = A_matrix_dev[index_sav];
    const double G        = G_matrix_dev[index_sav];
    const double H        = H_matrix_dev[index_sav];
    const double exptrm   = exptrm_dev[index_sav];
    const double lambda   = lambda_dev[index_sav];

    const double inv_u1 = 1.0 / ubar1;

    double term1 = xint_dev[index_read] * exp(-dtau * inv_u1);

    double term2 = (w0_og * f0 / (4.0 * pi)) * p_single * exp(-tau_og / ubar0) *
                   (1.0 - exp(-dtau_og * (ubar0 + ubar1) / (ubar0 * ubar1))) *
                   (ubar0 / (ubar0 + ubar1));

    double term3 = A *
                   (1.0 - exp(-dtau * (ubar0 + ubar1) / (ubar0 * ubar1))) *
                   (ubar0 / (ubar0 + ubar1));

    double term4 = G *
                   (exp(exptrm - dtau * inv_u1) - 1.0) /
                   (lambda * ubar1 - 1.0);

    double term5 = H *
                   (1.0 - exp(-exptrm - dtau * inv_u1)) /
                   (lambda * ubar1 + 1.0);

    xint_dev[index_sav] = term1 + term2 + term3 + term4 + term5;
}


__global__ void compute_disco(
    double * __restrict__ albedo_dev,
    double * __restrict__ xint_dev,
    double gweight, double tweight,
    int nwno)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    albedo_dev[index] += xint_dev[index] * gweight * tweight;
}


__global__ void final_albedo(
    double * __restrict__ albedo_dev,
    double * __restrict__ f0pi_dev,
    double cos_theta,
    int nwno)
{
    int index = global_idx_1d();
    if (index >= nwno) return;

    const double pi  = 3.141592653589793238463;
    const double sym_fac = 2.0 * pi;

    albedo_dev[index] = sym_fac * 0.5 * albedo_dev[index] / f0pi_dev[index] *
                        (cos_theta + 1.0);
}


__global__ void calculate_flux_minus_plus_first(
    double * __restrict__ flux_minus_dev,
    double * __restrict__ flux_plus_dev,
    double * __restrict__ positive_dev,
    double * __restrict__ negative_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ c_plus_up_dev,
    double * __restrict__ c_minus_up_dev,
    int nlayer, int nwno)
{
    int index = global_idx_1d();
    int total = nlayer * nwno;
    if (index >= total) return;

    const double pos = positive_dev[index];
    const double neg = negative_dev[index];
    const double g   = gama_dev[index];

    flux_minus_dev[index] = g * pos + neg + c_minus_up_dev[index];
    flux_plus_dev[index]  = pos + g * neg + c_plus_up_dev[index];
}


__global__ void calculate_flux_minus_plus_second(
    double * __restrict__ flux_minus_dev,
    double * __restrict__ flux_plus_dev,
    double * __restrict__ positive_dev,
    double * __restrict__ negative_dev,
    double * __restrict__ exptrm_positive_dev,
    double * __restrict__ exptrm_minus_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ c_plus_down_dev,
    double * __restrict__ c_minus_down_dev,
    int nlevel, int nwno)
{
    int index = global_idx_1d();
    int total = nlevel * nwno;
    if (index >= total) return;

    if (index >= (nlevel - 1) * nwno && index < nlevel * nwno) {
        int index_last = index - nwno;

        const double pos = positive_dev[index_last];
        const double neg = negative_dev[index_last];
        const double ep  = exptrm_positive_dev[index_last];
        const double em  = exptrm_minus_dev[index_last];
        const double g   = gama_dev[index_last];

        flux_minus_dev[index] = g * pos * ep + neg * em + c_minus_down_dev[index_last];
        flux_plus_dev[index]  = pos * ep + g * neg * em + c_plus_down_dev[index_last];
    }
}


__global__ void flux_minus_update(
    double * __restrict__ flux_minus_dev,
    double * __restrict__ f0pi_dev,
    double * __restrict__ tau_dev,
    double u0,
    int nlevel, int nwno)
{
    int index = global_idx_1d();
    int total = nlevel * nwno;
    if (index >= total) return;

    int index_1d = index % nwno;

    flux_minus_dev[index] += u0 * f0pi_dev[index_1d] * exp(-tau_dev[index] / u0);
}


__global__ void calculate_flux_minus_plus_midpt(
    double * __restrict__ flux_minus_midpt_dev,
    double * __restrict__ flux_plus_midpt_dev,
    double * __restrict__ positive_dev,
    double * __restrict__ negative_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ a_plus_dev,
    double * __restrict__ a_minus_dev,
    double * __restrict__ tau_dev,
    double * __restrict__ dtau_dev,
    double * __restrict__ f0pi_dev,
    double * __restrict__ exptrm_dev,
    double u0,
    int nlayer, int nwno)
{
    int index = global_idx_1d();
    int total = nlayer * nwno;
    if (index >= total) return;

    int index_1d = index % nwno;

    const double exp_mid  = exp(0.5 * exptrm_dev[index]);
    const double exp_inv  = 1.0 / exp_mid;

    double taumid = tau_dev[index] + 0.5 * dtau_dev[index];

    const double a_plus  = a_plus_dev[index];
    const double a_minus = a_minus_dev[index];

    const double c_plus_mid  = a_plus  * exp(-taumid / u0);
    const double c_minus_mid = a_minus * exp(-taumid / u0);

    const double pos = positive_dev[index];
    const double neg = negative_dev[index];
    const double g   = gama_dev[index];

    const double f0 = f0pi_dev[index_1d];

    flux_minus_midpt_dev[index] =
        g * pos * exp_mid + neg * exp_inv + c_minus_mid +
        u0 * f0 * exp(-taumid / u0);

    flux_plus_midpt_dev[index] =
        pos * exp_mid + g * neg * exp_inv + c_plus_mid;
}


__global__ void get_lvl_fluxes(
    double * __restrict__ flux_minus_dev,
    double * __restrict__ flux_plus_dev,
    double * __restrict__ flux_minus_midpt_dev,
    double * __restrict__ flux_plus_midpt_dev,
    double * __restrict__ flux_minus_all_dev,
    double * __restrict__ flux_plus_all_dev,
    double * __restrict__ flux_minus_midpt_all_dev,
    double * __restrict__ flux_plus_midpt_all_dev,
    int i_iter,
    int nlevel, int nwno)
{
    int index = global_idx_1d();
    int total = nlevel * nwno;
    if (index >= total) return;

    int index_save = index + i_iter * total;

    flux_minus_all_dev[index_save]      = flux_minus_dev[index];
    flux_plus_all_dev[index_save]       = flux_plus_dev[index];
    flux_minus_midpt_all_dev[index_save] = flux_minus_midpt_dev[index];
    flux_plus_midpt_all_dev[index_save]  = flux_plus_midpt_dev[index];
}


// // ============================================
// // Persistent GPU context for reflected solver
// // ============================================
struct ReflectedContext {
    bool   initialized = false;
    int    nlevel = 0;
    int    nlayer = 0;
    int    nwno   = 0;
    int    ng     = 0;
    int    nt     = 0;

    // Configuration constants
    // int    atm_surf_reflect = 0;
    int    single_phase     = 0;
    int    multi_phase      = 0;
    int    get_toa_intensity = 0;
    int    get_lvl_flux      = 0;
    double frac_a = 0.0;
    double frac_b = 0.0;
    double frac_c = 0.0;
    double constant_back    = 0.0;
    double constant_forward = 0.0;
    double toon_coefficients = 0.0;

    // Device arrays (static inputs)
    double *wno_dev      = nullptr;
    double *f0pi_dev     = nullptr;
    double *tau_dev      = nullptr;
    double *tau_og_dev   = nullptr;
    double *dtau_dev     = nullptr;
    double *w0_dev       = nullptr;
    double *cosb_dev     = nullptr;
    double *gcos2_dev    = nullptr;
    double *ftau_cld_dev = nullptr;
    double *ftau_ray_dev = nullptr;
    double *w0_og_dev    = nullptr;
    double *cosb_og_dev  = nullptr;
    double *dtau_og_dev  = nullptr;
    double *atm_surf_reflect_dev = nullptr;

    // Working arrays
    double *g1_dev   = nullptr;
    double *g2_dev   = nullptr;
    double *g3_dev   = nullptr;
    double *g4_dev   = nullptr;
    double *lambda_dev = nullptr;
    double *gama_dev   = nullptr;

    double *c_plus_up_dev    = nullptr;
    double *c_minus_up_dev   = nullptr;
    double *c_plus_down_dev  = nullptr;
    double *c_minus_down_dev = nullptr;

    double *b_surface_dev        = nullptr;
    double *exptrm_positive_dev  = nullptr;
    double *exptrm_minus_dev     = nullptr;
    double *exptrm_dev           = nullptr;

    double *A_odd_dev = nullptr;
    double *B_odd_dev = nullptr;
    double *C_odd_dev = nullptr;
    double *D_odd_dev = nullptr;

    double *e1_dev = nullptr;
    double *e2_dev = nullptr;
    double *e3_dev = nullptr;
    double *e4_dev = nullptr;

    double *positive_dev = nullptr;
    double *negative_dev = nullptr;

    double *AS_dev = nullptr;
    double *DS_dev = nullptr;
    double *XK_dev = nullptr;

    double *G_matrix_dev = nullptr;
    double *H_matrix_dev = nullptr;
    double *A_matrix_dev = nullptr;

    double *p_single_dev = nullptr;
    double *albedo_dev   = nullptr;

    double *xint_dev      = nullptr;
    double *xint_out_dev  = nullptr; // optional, but included per your list

    double *a_minus_dev = nullptr;
    double *a_plus_dev  = nullptr;

    // Flux arrays
    double *flux_minus_dev       = nullptr;
    double *flux_plus_dev        = nullptr;
    double *flux_minus_midpt_dev = nullptr;
    double *flux_plus_midpt_dev  = nullptr;

    double *flux_minus_all_dev       = nullptr;
    double *flux_plus_all_dev        = nullptr;
    double *flux_minus_midpt_all_dev = nullptr;
    double *flux_plus_midpt_all_dev  = nullptr;
};

static ReflectedContext g_ref_ctx;

// ---------- SET INPUTS: store device pointers directly ----------
extern "C" void get_reflected_1d_set_inputs(
    int    nlevel,
    const double *wno,       // device ptr from CuPy
    int    nwno,
    int    ng,
    int    nt,
    const double *dtau,      // device
    const double *tau,       // device
    const double *w0,        // device
    const double *cosb,      // device
    const double *gcos2,     // device
    const double *ftau_cld,  // device
    const double *ftau_ray,  // device
    const double *dtau_og,   // device
    const double *tau_og,    // device
    const double *w0_og,     // device
    const double *cosb_og,   // device
    const double *f0pi,      // device
    const double *atm_surf_reflect,  // device
    int    single_phase,
    int    multi_phase,
    double frac_a,
    double frac_b,
    double frac_c,
    double constant_back,
    double constant_forward,
    int    get_toa_intensity,
    int    get_lvl_flux,
    double toon_coefficients)
{
    ReflectedContext &ctx = g_ref_ctx;

    ctx.nlevel = nlevel;
    ctx.nlayer = nlevel - 1;
    ctx.nwno   = nwno;
    ctx.ng     = ng;
    ctx.nt     = nt;


    ctx.single_phase      = single_phase;
    ctx.multi_phase       = multi_phase;
    ctx.frac_a            = frac_a;
    ctx.frac_b            = frac_b;
    ctx.frac_c            = frac_c;
    ctx.constant_back     = constant_back;
    ctx.constant_forward  = constant_forward;
    ctx.get_toa_intensity = get_toa_intensity;
    ctx.get_lvl_flux      = get_lvl_flux;
    ctx.toon_coefficients = toon_coefficients;

    // *** Just store device pointers; NO cudaMalloc, NO cudaMemcpy ***
    ctx.wno_dev      = (double*)wno;
    ctx.f0pi_dev     = (double*)f0pi;
    ctx.tau_dev      = (double*)tau;
    ctx.tau_og_dev   = (double*)tau_og;
    ctx.dtau_dev     = (double*)dtau;
    ctx.w0_dev       = (double*)w0;
    ctx.cosb_dev     = (double*)cosb;
    ctx.gcos2_dev    = (double*)gcos2;
    ctx.ftau_cld_dev = (double*)ftau_cld;
    ctx.ftau_ray_dev = (double*)ftau_ray;
    ctx.w0_og_dev    = (double*)w0_og;
    ctx.cosb_og_dev  = (double*)cosb_og;
    ctx.dtau_og_dev  = (double*)dtau_og;
    ctx.atm_surf_reflect_dev  = (double*)atm_surf_reflect;

}


extern "C" void get_reflected_1d_allocate_buffers(
    int    nlevel,
    int    nwno,
    int    ng,
    int    nt)
{
    ReflectedContext &ctx = g_ref_ctx;

    if (ctx.initialized) {
        fprintf(stderr, "get_reflected_1d_init: already initialized, ignoring.\n");
        return;
    }

    ctx.nlevel = nlevel;
    ctx.nlayer = nlevel - 1;
    ctx.nwno   = nwno;
    ctx.ng     = ng;
    ctx.nt     = nt;

    int nlayer = ctx.nlayer;

    // Sizes
    size_t size_wno     = (size_t)nwno * sizeof(double);
    size_t size_layer   = (size_t)nlayer * nwno * sizeof(double);
    size_t size_level   = (size_t)nlevel * nwno * sizeof(double);
    size_t size_tridiag = (size_t)2 * nlayer * nwno * sizeof(double);
    // size_t size_flux_all = (size_t)ng * nt * nlevel * nwno * sizeof(double);
    size_t size_xint_out = (size_t)ng * nt * nwno * sizeof(double);


    // ---- Working arrays ----
    CUDA_CHECK(cudaMalloc(&ctx.g1_dev,   size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.g2_dev,   size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.g3_dev,   size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.g4_dev,   size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.lambda_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.gama_dev,   size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.c_plus_up_dev,    size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.c_minus_up_dev,   size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.c_plus_down_dev,  size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.c_minus_down_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.b_surface_dev,       size_wno));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_positive_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_minus_dev,    size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_dev,          size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.A_odd_dev, size_tridiag));
    CUDA_CHECK(cudaMalloc(&ctx.B_odd_dev, size_tridiag));
    CUDA_CHECK(cudaMalloc(&ctx.C_odd_dev, size_tridiag));
    CUDA_CHECK(cudaMalloc(&ctx.D_odd_dev, size_tridiag));

    CUDA_CHECK(cudaMalloc(&ctx.e1_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.e2_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.e3_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.e4_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.positive_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.negative_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.AS_dev, size_tridiag));
    CUDA_CHECK(cudaMalloc(&ctx.DS_dev, size_tridiag));
    CUDA_CHECK(cudaMalloc(&ctx.XK_dev, size_tridiag));

    CUDA_CHECK(cudaMalloc(&ctx.G_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.H_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.A_matrix_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.p_single_dev, size_layer));



    CUDA_CHECK(cudaMalloc(&ctx.xint_dev, size_level));
    CUDA_CHECK(cudaMalloc(&ctx.xint_out_dev, size_xint_out)); // optional use

    CUDA_CHECK(cudaMalloc(&ctx.a_minus_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.a_plus_dev,  size_layer));

    // Flux arrays
    CUDA_CHECK(cudaMalloc(&ctx.flux_minus_dev,       size_level));
    CUDA_CHECK(cudaMalloc(&ctx.flux_plus_dev,        size_level));
    CUDA_CHECK(cudaMalloc(&ctx.flux_minus_midpt_dev, size_level));
    CUDA_CHECK(cudaMalloc(&ctx.flux_plus_midpt_dev,  size_level));


    ctx.initialized = true;
}

extern "C" void get_reflected_1d_run(
    const double *ubar0,
    const double *ubar1,
    double        cos_theta,
    // double        surf_reflect,
    double        b_top,
    const double *gweight,
    const double *tweight,
    double *test_out,
    double *flux_minus_all_out,
    double *flux_plus_all_out,
    double *flux_minus_midpt_all_out,
    double *flux_plus_midpt_all_out)
{
    ReflectedContext &ctx = g_ref_ctx;
    if (!ctx.initialized) {
        fprintf(stderr, "get_reflected_1d_run called before init\n");
        return;
    }

    int nlevel = ctx.nlevel;
    int nlayer = ctx.nlayer;
    int nwno   = ctx.nwno;
    int ng     = ctx.ng;
    int nt     = ctx.nt;

    int N_layer  = nlayer * nwno;
    int N_level  = nlevel * nwno;



    ctx.albedo_dev = (double*)test_out;
    ctx.flux_minus_all_dev = (double*)flux_minus_all_out;
    ctx.flux_plus_all_dev = (double*)flux_plus_all_out;
    ctx.flux_minus_midpt_all_dev = (double*)flux_minus_midpt_all_out;
    ctx.flux_plus_midpt_all_dev = (double*)flux_plus_midpt_all_out;

    // ----------------------------
    // Kernel launch helpers
    // ----------------------------
    // const int BLOCK_SIZE = 256;
    // auto make_grid = [&](int N) {
    //     return dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // };

    // 1) Precompute g1,g2,lambda,gamma (depends only on static inputs)
    //    If geometry changes rarely, you could move this into init.
    calculate_g12_lg_quad<<<make_grid(N_layer), BLOCK_SIZE>>>(
        ctx.w0_dev, ctx.ftau_cld_dev, ctx.cosb_dev, N_layer,
        ctx.g1_dev, ctx.g2_dev, ctx.lambda_dev, ctx.gama_dev);

    // Reset albedo for this run
    CUDA_CHECK(cudaMemset(ctx.albedo_dev, 0, nwno * sizeof(double)));

    // 2) Loop over angular quadrature points
    for (int i_iter = 0; i_iter < ng * nt; ++i_iter) {

        // calculate_all
        calculate_all<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.w0_dev, ctx.ftau_cld_dev, ctx.cosb_dev, N_layer,
            ctx.g1_dev, ctx.g2_dev, ctx.lambda_dev,
            ctx.g3_dev, ctx.g4_dev,
            ctx.c_plus_up_dev, ctx.c_plus_down_dev,
            ctx.c_minus_up_dev, ctx.c_minus_down_dev,
            ctx.b_surface_dev,
            ubar0[i_iter],
            ctx.tau_dev, nwno,
            ctx.f0pi_dev, ctx.atm_surf_reflect_dev,
            ctx.dtau_dev, ctx.exptrm_dev,
            ctx.a_minus_dev, ctx.a_plus_dev);

        // exptrm -> exp+ / exp-
        calculate_allexptrm<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.exptrm_dev, ctx.exptrm_positive_dev,
            ctx.exptrm_minus_dev, 35, N_layer);

        // E matrix
        calculate_e_matrix<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.exptrm_positive_dev, ctx.exptrm_minus_dev,
            ctx.gama_dev,
            ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
            N_layer);

        // Tri-diagonal setup
        setup_tri_diag_all<<<make_grid(N_layer), BLOCK_SIZE>>>(
            nlayer, nwno,
            ctx.c_plus_up_dev, ctx.c_minus_up_dev,
            ctx.c_plus_down_dev, ctx.c_minus_down_dev,
            b_top, ctx.b_surface_dev, ctx.atm_surf_reflect_dev,
            ctx.gama_dev, ctx.dtau_dev,
            ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
            ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev);

        setup_tri_diag_1st<<<make_grid(nwno), BLOCK_SIZE>>>(
            nlayer, nwno,
            ctx.c_plus_up_dev, ctx.c_minus_up_dev,
            ctx.c_plus_down_dev, ctx.c_minus_down_dev,
            b_top, ctx.b_surface_dev, ctx.atm_surf_reflect_dev,
            ctx.gama_dev, ctx.dtau_dev,
            ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
            ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev);

        setup_tri_diag_last<<<make_grid(N_layer), BLOCK_SIZE>>>(
            nlayer, nwno,
            ctx.c_plus_up_dev, ctx.c_minus_up_dev,
            ctx.c_plus_down_dev, ctx.c_minus_down_dev,
            b_top, ctx.b_surface_dev, ctx.atm_surf_reflect_dev,
            ctx.gama_dev, ctx.dtau_dev,
            ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
            ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev);

        // Init tri-diagonal at bottom
        init_matrix<<<make_grid(nwno), BLOCK_SIZE>>>(
            ctx.AS_dev, ctx.DS_dev,
            ctx.A_odd_dev, ctx.B_odd_dev, ctx.D_odd_dev,
            nlayer, nwno);

        // Forward sweep
        for (int i_layer = 2 * nlayer - 2; i_layer > -1; --i_layer) {
            calculate_Xmatrix1<<<make_grid(nwno), BLOCK_SIZE>>>(
                ctx.AS_dev, ctx.DS_dev, ctx.XK_dev,
                ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev,
                nlayer, nwno, i_layer);
        }

        // Set XK bottom
        set_matrix_zero<<<make_grid(nwno), BLOCK_SIZE>>>(
            ctx.XK_dev, ctx.DS_dev,
            nlayer, nwno);

        // Back substitution
        for (int i_layer = 1; i_layer < 2 * nlayer; ++i_layer) {
            calculate_Xmatrix2<<<make_grid(nwno), BLOCK_SIZE>>>(
                ctx.AS_dev, ctx.DS_dev, ctx.XK_dev,
                nlayer, nwno, i_layer);
        }

        // Positive/negative
        calculate_pos_neg<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.XK_dev, ctx.positive_dev, ctx.negative_dev,
            nlayer, nwno);

        // Fluxes
        calculate_flux_minus_plus_first<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.flux_minus_dev, ctx.flux_plus_dev,
            ctx.positive_dev, ctx.negative_dev,
            ctx.gama_dev,
            ctx.c_plus_up_dev, ctx.c_minus_up_dev,
            nlayer, nwno);

        calculate_flux_minus_plus_second<<<make_grid(N_level), BLOCK_SIZE>>>(
            ctx.flux_minus_dev, ctx.flux_plus_dev,
            ctx.positive_dev, ctx.negative_dev,
            ctx.exptrm_positive_dev, ctx.exptrm_minus_dev,
            ctx.gama_dev,
            ctx.c_plus_down_dev, ctx.c_minus_down_dev,
            nlevel, nwno);

        flux_minus_update<<<make_grid(N_level), BLOCK_SIZE>>>(
            ctx.flux_minus_dev,
            ctx.f0pi_dev,
            ctx.tau_dev,
            ubar0[i_iter],
            nlevel, nwno);

        calculate_flux_minus_plus_midpt<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.flux_minus_midpt_dev, ctx.flux_plus_midpt_dev,
            ctx.positive_dev, ctx.negative_dev,
            ctx.gama_dev,
            ctx.a_plus_dev, ctx.a_minus_dev,
            ctx.tau_dev, ctx.dtau_dev,
            ctx.f0pi_dev, ctx.exptrm_dev,
            ubar0[i_iter],
            nlayer, nwno);

        get_lvl_fluxes<<<make_grid(N_level), BLOCK_SIZE>>>(
            ctx.flux_minus_dev, ctx.flux_plus_dev,
            ctx.flux_minus_midpt_dev, ctx.flux_plus_midpt_dev,
            ctx.flux_minus_all_dev, ctx.flux_plus_all_dev,
            ctx.flux_minus_midpt_all_dev, ctx.flux_plus_midpt_all_dev,
            i_iter, nlevel, nwno);

        // Intensity at TOA
        calculate_flux0<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.xint_dev,
            ctx.positive_dev, ctx.negative_dev,
            ctx.exptrm_positive_dev, ctx.exptrm_minus_dev,
            ctx.gama_dev, ctx.c_plus_down_dev,
            nlayer, nwno);

        // G/H/A matrices
        calculate_GHA<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.G_matrix_dev, ctx.H_matrix_dev, ctx.A_matrix_dev,
            ctx.ftau_cld_dev, ctx.cosb_dev, ctx.gcos2_dev,
            ctx.positive_dev, ctx.negative_dev,
            ctx.gama_dev, ctx.w0_dev,
            ctx.c_plus_up_dev, ctx.c_minus_up_dev,
            ubar1[i_iter],
            nlayer, nwno);

        // Direct scattering
        direct_scattering<<<make_grid(N_layer), BLOCK_SIZE>>>(
            ctx.constant_forward, ctx.constant_back,
            ctx.frac_a, ctx.frac_b, ctx.frac_c,
            cos_theta,
            ctx.cosb_og_dev, ctx.ftau_cld_dev, ctx.ftau_ray_dev,
            ctx.p_single_dev,
            nlayer, nwno);

        // Downward sweep for intensity
        for (int i_layer = nlayer - 1; i_layer > -1; --i_layer) {
            calculate_xint<<<make_grid(nwno), BLOCK_SIZE>>>(
                ctx.xint_dev,
                ctx.dtau_og_dev, ctx.dtau_dev,
                ctx.w0_og_dev, ctx.f0pi_dev, ctx.p_single_dev,
                ctx.tau_og_dev,
                ctx.A_matrix_dev, ctx.G_matrix_dev, ctx.H_matrix_dev,
                ctx.exptrm_dev, ctx.lambda_dev,
                ubar0[i_iter], ubar1[i_iter],
                i_layer, nwno);
        }

        // Integrate over angle
        compute_disco<<<make_grid(nwno), BLOCK_SIZE>>>(
            ctx.albedo_dev,
            ctx.xint_dev,
            gweight[i_iter], tweight[0], // if nt>1, adjust index
            nwno);
    }

    // Final albedo from albedo_dev
    final_albedo<<<make_grid(nwno), BLOCK_SIZE>>>(
        ctx.albedo_dev, ctx.f0pi_dev, cos_theta, nwno);


}

extern "C" void get_reflected_1d_free()
{
    ReflectedContext &ctx = g_ref_ctx;
    if (!ctx.initialized) return;

    auto FREE = [](double *&p) {
        if (p) { CUDA_CHECK(cudaFree(p)); p = nullptr; }
    };

    // Static inputs
    // FREE(ctx.wno_dev);
    // FREE(ctx.f0pi_dev);
    // FREE(ctx.tau_dev);
    // FREE(ctx.tau_og_dev);
    // FREE(ctx.dtau_dev);
    // FREE(ctx.w0_dev);
    // FREE(ctx.cosb_dev);
    // FREE(ctx.gcos2_dev);
    // FREE(ctx.ftau_cld_dev);
    // FREE(ctx.ftau_ray_dev);
    // FREE(ctx.w0_og_dev);
    // FREE(ctx.cosb_og_dev);
    // FREE(ctx.dtau_og_dev);

    // Working arrays
    FREE(ctx.g1_dev);
    FREE(ctx.g2_dev);
    FREE(ctx.g3_dev);
    FREE(ctx.g4_dev);
    FREE(ctx.lambda_dev);
    FREE(ctx.gama_dev);

    FREE(ctx.c_plus_up_dev);
    FREE(ctx.c_minus_up_dev);
    FREE(ctx.c_plus_down_dev);
    FREE(ctx.c_minus_down_dev);

    FREE(ctx.b_surface_dev);
    FREE(ctx.exptrm_positive_dev);
    FREE(ctx.exptrm_minus_dev);
    FREE(ctx.exptrm_dev);

    FREE(ctx.A_odd_dev);
    FREE(ctx.B_odd_dev);
    FREE(ctx.C_odd_dev);
    FREE(ctx.D_odd_dev);

    FREE(ctx.e1_dev);
    FREE(ctx.e2_dev);
    FREE(ctx.e3_dev);
    FREE(ctx.e4_dev);

    FREE(ctx.positive_dev);
    FREE(ctx.negative_dev);

    FREE(ctx.AS_dev);
    FREE(ctx.DS_dev);
    FREE(ctx.XK_dev);

    FREE(ctx.G_matrix_dev);
    FREE(ctx.H_matrix_dev);
    FREE(ctx.A_matrix_dev);

    FREE(ctx.p_single_dev);
    // FREE(ctx.albedo_dev);

    FREE(ctx.xint_dev);
    FREE(ctx.xint_out_dev);

    FREE(ctx.a_minus_dev);
    FREE(ctx.a_plus_dev);

    FREE(ctx.flux_minus_dev);
    FREE(ctx.flux_plus_dev);
    FREE(ctx.flux_minus_midpt_dev);
    FREE(ctx.flux_plus_midpt_dev);

    // FREE(ctx.flux_minus_all_dev);
    // FREE(ctx.flux_plus_all_dev);
    // FREE(ctx.flux_minus_midpt_all_dev);
    // FREE(ctx.flux_plus_midpt_all_dev);

    ctx.initialized = false;
}
