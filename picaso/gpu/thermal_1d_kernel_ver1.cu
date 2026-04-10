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


__global__ void calculate_e_matrix(double *exptrm_positive_dev, double *exptrm_minus_dev,double *gama_dev,double *e1_dev, double *e2_dev, double *e3_dev, double *e4_dev,int vec_size){

  // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  // int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < vec_size){

    e1_dev[index] = exptrm_positive_dev[index] + gama_dev[index]*exptrm_minus_dev[index];
    e2_dev[index] = exptrm_positive_dev[index] - gama_dev[index]*exptrm_minus_dev[index];
    e3_dev[index] = gama_dev[index]*exptrm_positive_dev[index] + exptrm_minus_dev[index];
    e4_dev[index] = gama_dev[index]*exptrm_positive_dev[index] - exptrm_minus_dev[index];


  }

}





__global__ void setup_tri_diag_1st(int nlayer,int nwno,  double *c_plus_up_dev, double *c_minus_up_dev,double *c_plus_down_dev, double *c_minus_down_dev,
                                  double *b_top, double *b_surface_dev, double *surf_reflect_dev,
                                 double *gama_dev, double *dtau_dev,  double *e1_dev,  double *e2_dev,double *e3_dev,  double *e4_dev,
                                  double *A_odd_dev, double *B_odd_dev, double *C_odd_dev, double *D_odd_dev){



   // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
   // int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   if ( index < nwno){

     A_odd_dev[index] = 0.0;
     B_odd_dev[index] = gama_dev[index]+1.0;
     C_odd_dev[index] =gama_dev[index]-1.0;
     D_odd_dev[index] = b_top[index]- c_minus_up_dev[index];

   }

}

__global__ void setup_tri_diag_last(int nlayer,int nwno,  double *c_plus_up_dev, double *c_minus_up_dev,double *c_plus_down_dev, double *c_minus_down_dev,
                                  double *b_top, double *b_surface_dev, double *surf_reflect_dev,
                                 double *gama_dev, double *dtau_dev,  double *e1_dev,  double *e2_dev,double *e3_dev,  double *e4_dev,
                                  double *A_odd_dev, double *B_odd_dev, double *C_odd_dev, double *D_odd_dev){


   //
   // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
   // int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   if ( (index > (nlayer-1)*nwno) && (index < nlayer*nwno) ){

     A_odd_dev[index+nlayer*nwno] = e1_dev[index]-surf_reflect_dev[index -(nlayer-1)*nwno ]*e3_dev[index];
     B_odd_dev[index+nlayer*nwno] = e2_dev[index]-surf_reflect_dev[index -(nlayer-1)*nwno ]*e4_dev[index];
     C_odd_dev[index+nlayer*nwno] = 0.0;
     D_odd_dev[index+nlayer*nwno] = b_surface_dev[index -(nlayer-1)*nwno ]-c_plus_down_dev[index] + surf_reflect_dev[index -(nlayer-1)*nwno ]*c_minus_down_dev[index];

   }

}




__global__ void setup_tri_diag_all(int nlayer,int nwno,  double *c_plus_up_dev, double *c_minus_up_dev,double *c_plus_down_dev, double *c_minus_down_dev,
                                  double *b_top, double *b_surface_dev, double *surf_reflect_dev,
                                 double *gama_dev, double *dtau_dev,  double *e1_dev,  double *e2_dev,double *e3_dev,  double *e4_dev,
                                  double *A_odd_dev, double *B_odd_dev, double *C_odd_dev, double *D_odd_dev){



   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int index_wn = index % nwno;
   int index_layer = index / nwno;


   if ( (index+nwno) < nlayer*nwno){

     int index_odd_sav = nwno*(index_layer*2+1)+index_wn;

     A_odd_dev[index_odd_sav] = (e1_dev[index]+e3_dev[index]) * (gama_dev[index+nwno]-1.0);

     B_odd_dev[index_odd_sav] = (e2_dev[index]+e4_dev[index]) * (gama_dev[index+nwno]-1.0);

     C_odd_dev[index_odd_sav] = 2.0 * (1.0- pow(gama_dev[index+nwno],2));

     D_odd_dev[index_odd_sav] =((gama_dev[index+nwno]-1.0)*(c_plus_up_dev[index+nwno] - c_plus_down_dev[index])
                              + (1.0-gama_dev[index+nwno])*(c_minus_down_dev[index] - c_minus_up_dev[index+nwno]));


    int index_even_sav = nwno*(index_layer*2+2)+index_wn;

    A_odd_dev[index_even_sav] = 2.0*(1.0-pow(gama_dev[index],2));

    B_odd_dev[index_even_sav] =  (e1_dev[index]-e3_dev[index]) * (gama_dev[index+nwno]+1.0);

    C_odd_dev[index_even_sav] = (e1_dev[index]+e3_dev[index]) * (gama_dev[index+nwno]-1.0);

    D_odd_dev[index_even_sav] =(e3_dev[index]*(c_plus_up_dev[index+nwno] - c_plus_down_dev[index]) +
                            e1_dev[index]*(c_minus_down_dev[index] - c_minus_up_dev[index+nwno]));



   }

}





__global__ void calculate_blackbody(double *lvl_T_dev, double *wno_dev, double *wno0_dev, double *all_b_dev,
                                    int nlevel, int nwno, int calc_type) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int index_wn = index % nwno;
    int index_layer = index / nwno;

    double h = 6.62607004e-27;
    double c = 2.99792458e+10;
    double k = 1.38064852e-16;
    double c1 = 2 * h * pow(c, 2);
    double c2 = h * c / k;

    if (index < nlevel * nwno) {
        if (calc_type == 0) {
            // Blackbody calculation
            double wn_inver = 1.0 / wno_dev[index_wn];
            all_b_dev[index] = (2.0 * h * pow(c, 2)) / (pow(wn_inver, 5)) / (exp((h * c) / (lvl_T_dev[index_layer] * wn_inver * k)) - 1.0);
        } else if (calc_type == 1) {
            // Blackbody integrated calculation
            int nbb = 1; // Assuming nbb is 1; adjust as needed
            for (int i_iter = -nbb; i_iter < nbb + 1; i_iter++) {
                double wavenum = wno_dev[index_wn] + i_iter * wno0_dev[index_wn] / (2.0 * nbb);
                all_b_dev[index] += c1 * pow(wavenum, 3) / (exp(c2 * wavenum / lvl_T_dev[index_layer]) - 1.0);
            }
            all_b_dev[index] /= (2 * nbb + 1.0);
        }
    }
}




__global__ void initialize_parameters(
    const double * __restrict__ all_b_dev,
    double * __restrict__ b0_dev,
    double * __restrict__ b1_dev,
    const double * __restrict__ dtau_og_dev,
    const double * __restrict__ w0_no_raman_dev,
    const double * __restrict__ cosb_og_dev,
    double * __restrict__ alpha_dev,
    double * __restrict__ lamda_dev,
    double * __restrict__ gama_dev,
    double * __restrict__ g1_plus_g2_dev,
    double * __restrict__ c_plus_up_dev,
    double * __restrict__ c_minus_up_dev,
    double * __restrict__ c_plus_down_dev,
    double * __restrict__ c_minus_down_dev,
    const double * __restrict__ lvl_T_dev,  // not used but kept for interface
    const double * __restrict__ lvl_P_dev,
    double * __restrict__ exptrm_dev,
    double * __restrict__ exptrm_positive_dev,
    double * __restrict__ exptrm_minus_dev,
    double * __restrict__ tau_top_dev,
    double * __restrict__ b_top_dev,
    double * __restrict__ b_surface_dev,
    double * __restrict__ surf_reflect_dev, //neb-q does this need a __restrict? 
    double mu1,
    double hard_surface,
    int nlevel,
    int nwno)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int nlayer = nlevel - 1;
    int n_layer_tot = nlayer * nwno;

    constexpr double pi  = 3.14159265358979323846;
    // h, c, k are not needed here

    // ----------------------------------------
    // Part 1: layer-based work (idx < (nlevel-1)*nwno)
    // ----------------------------------------
    if (idx < n_layer_tot) {
        // flatten
        // int layer = idx / nwno;
        // int wn    = idx % nwno;  // unused

        // b0, b1
        double b0   = all_b_dev[idx];
        double dtau = dtau_og_dev[idx];
        double b1   = (all_b_dev[idx + nwno] - b0) / dtau;

        b0_dev[idx] = b0;
        b1_dev[idx] = b1;

        // g1, g2, alpha, lamda, gama, g1_plus_g2
        double w0   = w0_no_raman_dev[idx];
        double cosb = cosb_og_dev[idx];

        double g1 = 2.0 - w0 * (1.0 + cosb);
        double g2 = w0 * (1.0 - cosb);

        double alpha_val = sqrt((1.0 - w0) /
                                (1.0 - w0 * cosb));
        double lamda_val = sqrt(g1 * g1 - g2 * g2);
        double gama_val  = (g1 - lamda_val) / g2;
        double g1pg2_inv = 1.0 / (g1 + g2);

        alpha_dev[idx]       = alpha_val;
        lamda_dev[idx]       = lamda_val;
        gama_dev[idx]        = gama_val;
        g1_plus_g2_dev[idx]  = g1pg2_inv;

        // C-matrix terms
        double common_up    = 2.0 * pi * mu1;
        double common_down  = 2.0 * pi * mu1;

        double s1 = b0 + b1 * g1pg2_inv;
        double s2 = b0 - b1 * g1pg2_inv;
        double s3 = b0 + b1 * dtau + b1 * g1pg2_inv;
        double s4 = b0 + b1 * dtau - b1 * g1pg2_inv;

        c_plus_up_dev[idx]    = common_up   * s1;
        c_minus_up_dev[idx]   = common_up   * s2;
        c_plus_down_dev[idx]  = common_down * s3;
        c_minus_down_dev[idx] = common_down * s4;

        // exptrm
        double lam_dtau = lamda_val * dtau;
        double val = (lam_dtau > 35.0) ? 35.0 : lam_dtau;   // avoid overflow
        exptrm_dev[idx]            = val;
        double epos                = exp(val);
        exptrm_positive_dev[idx]   = epos;
        exptrm_minus_dev[idx]      = 1.0 / epos;
    }

    // ----------------------------------------
    // Part 2: top / bottom boundary work (idx < nwno)
    // ----------------------------------------
    if (idx < nwno) {
        // top boundary
        double P0 = lvl_P_dev[0];
        double P1 = lvl_P_dev[1];
        double dtau0 = dtau_og_dev[idx];  // layer 0, same idx

        double tau_top = dtau0 * P0 / (P1 - P0);
        tau_top_dev[idx] = tau_top;

        double b_top = (1.0 - exp(-tau_top / mu1)) * all_b_dev[idx] * pi;
        b_top_dev[idx] = b_top;

        // bottom boundary
        int idx_allb = idx + (nlevel - 1) * nwno;
        int idx_b    = idx + (nlevel - 2) * nwno;

        double b_surf;
        if (hard_surface == 1.0) {
            b_surf = (1.0 - surf_reflect_dev[idx]) * all_b_dev[idx_b] * pi;
        } else {
            b_surf = (all_b_dev[idx_allb] + b1_dev[idx_b] * mu1) * pi;
        }
        b_surface_dev[idx] = b_surf;
    }
}




__global__ void initialize_and_solve_tridiagonal(
    double * __restrict__ AS_dev,
    double * __restrict__ DS_dev,
    double * __restrict__ XK_dev,
    const double * __restrict__ A_odd_dev,
    const double * __restrict__ B_odd_dev,
    const double * __restrict__ C_odd_dev,
    const double * __restrict__ D_odd_dev,
    int nlayer,      // this should be (nlevel-1) as you pass it now
    int nwno)
{
    int wn = blockIdx.x * blockDim.x + threadIdx.x;
    if (wn >= nwno) return;

    // NOTE: this matches your original "2*(nlayer-1)" behavior
    int total_layers = 2 * (nlayer - 1);
    int stride = nwno;

    // --------------------------
    // Backward sweep: compute AS, DS
    // --------------------------
    int last_row = total_layers - 1;
    int idx_last = last_row * stride + wn;

    double B_last = B_odd_dev[idx_last] + 1e-16;
    double invB   = 1.0 / B_last;

    AS_dev[idx_last] = A_odd_dev[idx_last] * invB;
    DS_dev[idx_last] = D_odd_dev[idx_last] * invB;

    for (int row = total_layers - 2; row >= 0; --row) {
        int idx      = row * stride + wn;
        int idx_next = idx + stride;

        double Bv = B_odd_dev[idx] - C_odd_dev[idx] * AS_dev[idx_next];
        double inv = 1.0 / (Bv + 1e-16);

        AS_dev[idx] = A_odd_dev[idx] * inv;
        DS_dev[idx] = (D_odd_dev[idx] - C_odd_dev[idx] * DS_dev[idx_next]) * inv;
    }

    // --------------------------
    // Forward sweep: compute solution XK
    // --------------------------
    int idx0 = wn;  // row 0 * stride + wn
    XK_dev[idx0] = DS_dev[idx0];

    for (int row = 1; row < total_layers; ++row) {
        int idx  = row * stride + wn;
        int iprev = idx - stride;
        XK_dev[idx] = DS_dev[idx] - AS_dev[idx] * XK_dev[iprev];
    }
}


__global__ void calculate_matrices_and_exponents(
    const double * __restrict__ XK_dev,
    double * __restrict__ positive_dev,
    double * __restrict__ negative_dev,
    const double * __restrict__ lamda_dev,
    const double * __restrict__ gama_dev,
    const double * __restrict__ b0_dev,
    const double * __restrict__ b1_dev,
    const double * __restrict__ g1_plus_g2_dev,
    double mu1,
    double * __restrict__ G_matrix_dev,
    double * __restrict__ H_matrix_dev,
    double * __restrict__ J_matrix_dev,
    double * __restrict__ K_matrix_dev,
    double * __restrict__ alpha1_matrix_dev,
    double * __restrict__ alpha2_matrix_dev,
    double * __restrict__ sigma1_matrix_dev,
    double * __restrict__ sigma2_matrix_dev,
    const double * __restrict__ exptrm_dev,
    double * __restrict__ exptrm_positive_mdpt_dev,
    double * __restrict__ exptrm_minus_mdpt_dev,
    int nlevel,
    int nwno)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nlayer = nlevel - 1;
    int total = nlayer * nwno;

    if (idx >= total) return;

    int layer = idx / nwno;
    int wn    = idx % nwno;

    int idx_odd  = (layer * 2 + 1) * nwno + wn;
    int idx_even = (layer * 2    ) * nwno + wn;

    // Step 1: positive / negative from XK
    double x_even = XK_dev[idx_even];
    double x_odd  = XK_dev[idx_odd];

    double pos = x_even + x_odd;
    double neg = x_even - x_odd;

    positive_dev[idx] = pos;
    negative_dev[idx] = neg;

    // Step 2: matrices
    double lam    = lamda_dev[idx];
    double gamma  = gama_dev[idx];
    double b0     = b0_dev[idx];
    double b1     = b1_dev[idx];
    double g1g2   = g1_plus_g2_dev[idx];

    constexpr double pi = 3.14159265358979323846;
    double inv_mu1 = 1.0 / mu1;

    double lam_plus = lam + inv_mu1;
    double lam_minus = inv_mu1 - lam;

    // G,H,J,K
    G_matrix_dev[idx] = lam_minus * pos;
    H_matrix_dev[idx] = gamma * lam_plus * neg;
    J_matrix_dev[idx] = gamma * lam_plus * pos;
    K_matrix_dev[idx] = lam_minus * neg;

    // alpha / sigma
    // alpha1 = 2*pi*(b0 + b1*(g1_plus_g2 - mu1))
    // sigma1 = 2*pi*(b0 - b1*(g1_plus_g2 - mu1))
    double shift = g1g2 - mu1;
    double common = 2.0 * pi;

    double a1 = b0 + b1 * shift;
    double s1 = b0 - b1 * shift;

    alpha1_matrix_dev[idx] = common * a1;
    alpha2_matrix_dev[idx] = common * b1;
    sigma1_matrix_dev[idx] = common * s1;
    sigma2_matrix_dev[idx] = common * b1;

    // Step 3: midpoint exponentials
    double e_val = exptrm_dev[idx];
    double half  = 0.5 * e_val;
    double epos  = exp(half);
    exptrm_positive_mdpt_dev[idx] = epos;
    exptrm_minus_mdpt_dev[idx]    = 1.0 / epos;
}






__global__ void fill_flux_plus_minus(double *all_b_dev, double *b1_dev, double *tau_top_dev, double *surf_reflect_dev, double ubar1,double *flux_plus_dev, double *flux_minus_dev, double hard_surface,
                                    int nlevel, int nwno){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int index_allb = index + (nlevel - 1)* nwno;
  int index_out = index + (nlevel - 2)* nwno;
  // int index_out_flux = index + (nlevel - 1)* nwno;

  double pi  =3.141592653589793238463;

  if ( (index >= 0) && (index < nwno) ){
    if (hard_surface == 1.0){
      flux_plus_dev[index_allb] = (1.0 - surf_reflect_dev[index]) * all_b_dev[index_allb]*2*pi;

    }else{
      flux_plus_dev[index_allb] = (all_b_dev[index_allb] + b1_dev[index_out]*ubar1)*2*pi;
    }

    flux_minus_dev[index] = (1 - exp(-1*tau_top_dev[index] / ubar1)) * all_b_dev[index] *2*pi;


  }

}



__global__ void fill_exptrm_angle(double *dtau_og_dev, double *exptrm_angle_dev, double *exptrm_angle_mdpt_dev, double ubar1, int nlevel, int nwno){

  // int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  // int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if ( (index >= 0) && (index < nwno*(nlevel - 1) ) ){

    exptrm_angle_dev[index] = exp( -1* dtau_og_dev[index] / ubar1);
    exptrm_angle_mdpt_dev[index] = exp( -0.5 * dtau_og_dev[index] / ubar1);

  }

}


__global__ void update_flux_combined_one_kernel_loop_new(
    double * __restrict__ flux_minus_dev,
    double * __restrict__ flux_minus_mdpt_dev,
    double * __restrict__ flux_plus_dev,
    double * __restrict__ flux_plus_mdpt_dev,
    const double * __restrict__ exptrm_angle_dev,
    const double * __restrict__ exptrm_angle_mdpt_dev,
    const double * __restrict__ J_matrix_dev,
    const double * __restrict__ G_matrix_dev,
    const double * __restrict__ lamda_dev,
    const double * __restrict__ exptrm_positive_dev,
    const double * __restrict__ exptrm_positive_mdpt_dev,
    const double * __restrict__ K_matrix_dev,
    const double * __restrict__ H_matrix_dev,
    const double * __restrict__ exptrm_minus_dev,
    const double * __restrict__ exptrm_minus_mdpt_dev,
    const double * __restrict__ sigma1_matrix_dev,
    const double * __restrict__ sigma2_matrix_dev,
    const double * __restrict__ alpha1_matrix_dev,
    const double * __restrict__ alpha2_matrix_dev,
    const double * __restrict__ dtau_og_dev,
    double ubar1,
    int nlevel,
    int nwno)
{
    int wn = blockIdx.x * blockDim.x + threadIdx.x;
    if (wn >= nwno) return;

    // Loop over layers for this wavenumber
    for (int itop = 0; itop < nlevel - 1; ++itop) {
        int ibot = nlevel - 2 - itop;

        // Flattened indices
        int idx_itop      = itop * nwno + wn;
        int idx_itop_out  = (itop + 1) * nwno + wn;
        int idx_ibot      = ibot * nwno + wn;
        int idx_ibot_out  = (ibot + 1) * nwno + wn;

        // --- cache per-layer values (reduces loads & repeated ops) ---
        double lam_itop     = lamda_dev[idx_itop];
        double dtau_itop    = dtau_og_dev[idx_itop];
        double lam_ibar     = lamda_dev[idx_ibot];
        double dtau_ibot    = dtau_og_dev[idx_ibot];

        double lamubar_itop = lam_itop * ubar1;
        double lamubar_ibot = lam_ibar * ubar1;

        double exp_ang_itop      = exptrm_angle_dev[idx_itop];
        double exp_ang_mdpt_itop = exptrm_angle_mdpt_dev[idx_itop];
        double exp_ang_ibot      = exptrm_angle_dev[idx_ibot];
        double exp_ang_mdpt_ibot = exptrm_angle_mdpt_dev[idx_ibot];

        double exp_pos_itop      = exptrm_positive_dev[idx_itop];
        double exp_pos_mdpt_itop = exptrm_positive_mdpt_dev[idx_itop];
        double exp_neg_itop      = exptrm_minus_dev[idx_itop];
        double exp_neg_mdpt_itop = exptrm_minus_mdpt_dev[idx_itop];

        double exp_pos_ibot      = exptrm_positive_dev[idx_ibot];
        double exp_pos_mdpt_ibot = exptrm_positive_mdpt_dev[idx_ibot];
        double exp_neg_ibot      = exptrm_minus_dev[idx_ibot];
        double exp_neg_mdpt_ibot = exptrm_minus_mdpt_dev[idx_ibot];

        double J_itop  = J_matrix_dev[idx_itop];
        double K_itop  = K_matrix_dev[idx_itop];
        double sig1_it = sigma1_matrix_dev[idx_itop];
        double sig2_it = sigma2_matrix_dev[idx_itop];

        double G_ibot  = G_matrix_dev[idx_ibot];
        double H_ibot  = H_matrix_dev[idx_ibot];
        double alp1_ib = alpha1_matrix_dev[idx_ibot];
        double alp2_ib = alpha2_matrix_dev[idx_ibot];

        // Precompute denominators (small but easy win)
        double denom_Jm = 1.0 / (lamubar_itop + 1.0);
        double denom_Km = 1.0 / (lamubar_itop - 1.0);
        double denom_Gp = 1.0 / (lamubar_ibot - 1.0);
        double denom_Hp = 1.0 / (lamubar_ibot + 1.0);

        // ----------------------------------------------------------------
        // Step 1: update_flux_minus (itop → itop+1)
        // ----------------------------------------------------------------
        {
            double f_in = flux_minus_dev[idx_itop];

            double termJ = J_itop * denom_Jm * (exp_pos_itop - exp_ang_itop);
            double termK = K_itop * denom_Km * (exp_ang_itop   - exp_neg_itop);
            double termS = sig1_it * (1.0 - exp_ang_itop)
                         + sig2_it * (ubar1 * exp_ang_itop + dtau_itop - ubar1);

            flux_minus_dev[idx_itop_out] = f_in * exp_ang_itop + termJ + termK + termS;
        }

        // ----------------------------------------------------------------
        // Step 2: update_flux_minus_mdpt (itop midpoint)
        // ----------------------------------------------------------------
        {
            double f_in = flux_minus_dev[idx_itop];  // same as original code

            double termJ = J_itop * denom_Jm * (exp_pos_mdpt_itop - exp_ang_mdpt_itop);
            double termK = K_itop * (1.0 / (-lamubar_itop + 1.0)) *
                           (exp_neg_mdpt_itop - exp_ang_mdpt_itop);

            double termS = sig1_it * (1.0 - exp_ang_mdpt_itop)
                         + sig2_it * (ubar1 * exp_ang_mdpt_itop + 0.5 * dtau_itop - ubar1);

            flux_minus_mdpt_dev[idx_itop] = f_in * exp_ang_mdpt_itop + termJ + termK + termS;
        }

        // ----------------------------------------------------------------
        // Step 3: update_flux_plus (ibot+1 → ibot)
        // ----------------------------------------------------------------
        {
            double f_in = flux_plus_dev[idx_ibot_out];

            double termG = G_ibot * denom_Gp * (exp_pos_ibot * exp_ang_ibot - 1.0);
            double termH = H_ibot * denom_Hp * (1.0 - exp_neg_ibot * exp_ang_ibot);

            double termS = alp1_ib * (1.0 - exp_ang_ibot)
                         + alp2_ib * (ubar1 - (dtau_ibot + ubar1) * exp_ang_ibot);

            flux_plus_dev[idx_ibot] = f_in * exp_ang_ibot + termG + termH + termS;
        }

        // ----------------------------------------------------------------
        // Step 4: update_flux_plus_mdpt (ibot midpoint)
        // ----------------------------------------------------------------
        {
            double f_in = flux_plus_dev[idx_ibot_out];

            double termG = G_ibot * denom_Gp *
                           (exp_pos_ibot * exp_ang_mdpt_ibot - exp_pos_mdpt_ibot);

            double termH = - H_ibot * denom_Hp *
                           (exp_neg_ibot * exp_ang_mdpt_ibot - exp_neg_mdpt_ibot);

            double termS = alp1_ib * (1.0 - exp_ang_mdpt_ibot)
                         + alp2_ib * (ubar1 + 0.5 * dtau_ibot
                                     - (dtau_ibot + ubar1) * exp_ang_mdpt_ibot);

            flux_plus_mdpt_dev[idx_ibot] = f_in * exp_ang_mdpt_ibot + termG + termH + termS;
        }
    }
}


__global__ void get_lvl_fluxes(double *flux_minus_dev, double *flux_plus_dev, double *flux_minus_mdpt_dev, double *flux_plus_mdpt_dev,
                                          double *flux_minus_all_dev, double *flux_plus_all_dev, double *flux_minus_mdpt_all_dev, double *flux_plus_mdpt_all_dev,
                                          int i_iter, int nlevel, int nwno){


   int blockId = blockIdx.x + blockIdx.y * gridDim.x;
   int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   int index_save = index + i_iter*nlevel*nwno;

   if ( (index >= 0) && (index < nlevel*nwno) ){

     flux_minus_all_dev[index_save] = flux_minus_dev[index];
     flux_plus_all_dev[index_save] = flux_plus_dev[index];
     flux_minus_mdpt_all_dev[index_save] = flux_minus_mdpt_dev[index];
     flux_plus_mdpt_all_dev[index_save] = flux_plus_mdpt_dev[index];

   }

}



__global__ void get_fluxes_at_top(double *flux_at_top_dev, double *flux_plus_mdpt_dev,int i_iter, int nlevel, int nwno){


   int blockId = blockIdx.x + blockIdx.y * gridDim.x;
   int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
   int index_save = index + i_iter*nwno;

   if ( (index >= 0) && (index < nwno) ){


     flux_at_top_dev[index_save] = flux_plus_mdpt_dev[index];

   }

}


// ============================================
// Persistent GPU context
// ============================================
struct ThermalContext {
    bool   initialized = false;
    int    nlevel = 0;
    int    nlayer = 0;
    int    nwno   = 0;
    int    ng     = 0;
    int    nt     = 0;

    // Device arrays (mirror what you had in get_thermal_1d)
    double *wno_dev = nullptr;
    double *wno0_dev = nullptr;
    double *dtau_og_dev = nullptr;
    double *w0_no_raman_dev = nullptr;
    double *cosb_og_dev = nullptr;
    double *lvl_T_dev = nullptr;
    double *lvl_P_dev = nullptr;
    double *surf_reflect_dev = nullptr;

    double *alpha_dev = nullptr;
    double *lamda_dev = nullptr;
    double *gama_dev = nullptr;
    double *c_plus_up_dev = nullptr;
    double *c_minus_up_dev = nullptr;
    double *c_plus_down_dev = nullptr;
    double *c_minus_down_dev = nullptr;
    double *b_surface_dev = nullptr;
    double *exptrm_positive_dev = nullptr;
    double *exptrm_minus_dev = nullptr;
    double *exptrm_dev = nullptr;

    double *all_b_dev = nullptr;
    double *b0_dev = nullptr;
    double *b1_dev = nullptr;
    double *g1_plus_g2_dev = nullptr;

    double *tau_top_dev = nullptr;
    double *b_top_dev = nullptr;

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
    double *J_matrix_dev = nullptr;
    double *K_matrix_dev = nullptr;
    double *alpha1_matrix_dev = nullptr;
    double *alpha2_matrix_dev = nullptr;
    double *sigma1_matrix_dev = nullptr;
    double *sigma2_matrix_dev = nullptr;

    double *exptrm_positive_mdpt_dev = nullptr;
    double *exptrm_minus_mdpt_dev = nullptr;

    double *flux_minus_dev = nullptr;
    double *flux_plus_dev = nullptr;
    double *flux_minus_mdpt_dev = nullptr;
    double *flux_plus_mdpt_dev = nullptr;

    double *flux_minus_all_dev = nullptr;
    double *flux_plus_all_dev = nullptr;
    double *flux_minus_mdpt_all_dev = nullptr;
    double *flux_plus_mdpt_all_dev = nullptr;

    double *flux_at_top_dev = nullptr;

    double *exptrm_angle_dev = nullptr;
    double *exptrm_angle_mdpt_dev = nullptr;
};

static ThermalContext g_ctx;


extern "C" void get_thermal_1d_set_inputs(
    int    nlevel,
    const double *wno,
    int    nwno,
    int    ng,
    int    nt,
    const double *dtau_og,
    const double *w0_no_raman,
    const double *cosb_og,
    const double *lvl_P,
    const double *lvl_T,
    const double *wno0)
{
    ThermalContext &ctx = g_ctx;

    ctx.nlevel = nlevel;
    ctx.nlayer = nlevel - 1;
    ctx.nwno   = nwno;
    ctx.ng     = ng;
    ctx.nt     = nt;

    // ---- Save ONLY device pointers (zero-copy) ----
    ctx.wno_dev        = (double*)wno;
    ctx.wno0_dev       = (double*)wno0;
    ctx.dtau_og_dev    = (double*)dtau_og;
    ctx.w0_no_raman_dev = (double*)w0_no_raman;
    ctx.cosb_og_dev    = (double*)cosb_og;
    ctx.lvl_P_dev      = (double*)lvl_P;
    ctx.lvl_T_dev      = (double*)lvl_T;

}

extern "C" void get_thermal_1d_allocate_buffers(
    int    nlevel,
    int    nwno,
    int    ng,
    int    nt)
{
    ThermalContext &ctx = g_ctx;

    if (ctx.initialized) {
        // already initialized – could free and re-init, or just return
        fprintf(stderr, "get_thermal_1d_init: already initialized, ignoring.\n");
        return;
    }

    ctx.nlevel = nlevel;
    ctx.nlayer = nlevel - 1;
    ctx.nwno   = nwno;
    ctx.ng     = ng;
    ctx.nt     = nt;

    int nlayer = ctx.nlayer;

    // Shorthand sizes
    size_t size_wno      = nwno * sizeof(double);
    size_t size_layer    = (size_t)nlayer * nwno * sizeof(double);
    size_t size_all_b    = (size_t)nlevel * nwno * sizeof(double);
    size_t size_flux_lvl = (size_t)nlevel * nwno * sizeof(double);

    // ---- Allocate all intermediate / output device arrays (no copies yet) ----
    CUDA_CHECK(cudaMalloc(&ctx.alpha_dev,      size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.lamda_dev,      size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.gama_dev,       size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.c_plus_up_dev,  size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.c_minus_up_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.c_plus_down_dev,size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.c_minus_down_dev,size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.b_surface_dev,  size_wno));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_positive_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_minus_dev,    size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_dev,          size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.all_b_dev,  size_all_b));
    CUDA_CHECK(cudaMalloc(&ctx.b0_dev,     size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.b1_dev,     size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.g1_plus_g2_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.tau_top_dev, size_wno));
    CUDA_CHECK(cudaMalloc(&ctx.b_top_dev,   size_wno));

    CUDA_CHECK(cudaMalloc(&ctx.A_odd_dev,  2 * size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.B_odd_dev,  2 * size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.C_odd_dev,  2 * size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.D_odd_dev,  2 * size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.e1_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.e2_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.e3_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.e4_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.positive_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.negative_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.AS_dev, 2 * size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.DS_dev, 2 * size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.XK_dev, 2 * size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.G_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.H_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.J_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.K_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.alpha1_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.alpha2_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.sigma1_matrix_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.sigma2_matrix_dev, size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.exptrm_positive_mdpt_dev, size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_minus_mdpt_dev,    size_layer));

    CUDA_CHECK(cudaMalloc(&ctx.flux_minus_dev,      size_flux_lvl));
    CUDA_CHECK(cudaMalloc(&ctx.flux_plus_dev,       size_flux_lvl));
    CUDA_CHECK(cudaMalloc(&ctx.flux_minus_mdpt_dev, size_flux_lvl));
    CUDA_CHECK(cudaMalloc(&ctx.flux_plus_mdpt_dev,  size_flux_lvl));


    CUDA_CHECK(cudaMalloc(&ctx.exptrm_angle_dev,      size_layer));
    CUDA_CHECK(cudaMalloc(&ctx.exptrm_angle_mdpt_dev, size_layer));

    ctx.initialized = true;
}

extern "C" void get_thermal_1d_run(
    const double *ubar1,
    double hard_surface,
    int    calc_type,
    double *test_out,
    double *flux_minus_all_out,
    double *flux_plus_all_out,
    double *flux_minus_mdpt_all_out,
    double *flux_plus_mdpt_all_out)
{
    ThermalContext &ctx = g_ctx;
    if (!ctx.initialized) {
        fprintf(stderr, "get_thermal_1d_run called before init\n");
        return;
    }

    int nlevel = ctx.nlevel;
    int nlayer = ctx.nlayer;
    int nwno   = ctx.nwno;
    int ng     = ctx.ng;
    int nt     = ctx.nt;

    ctx.flux_at_top_dev = (double*)test_out;
    ctx.flux_minus_all_dev = (double*)flux_minus_all_out;
    ctx.flux_plus_all_dev = (double*)flux_plus_all_out;
    ctx.flux_minus_mdpt_all_dev = (double*)flux_minus_mdpt_all_out;
    ctx.flux_plus_mdpt_all_dev = (double*)flux_plus_mdpt_all_out;


    // ----------------------------
    // Kernel launch parameters
    // ----------------------------
    int blockSize = 256;
    dim3 grid_all(   (nlevel * nwno        + blockSize - 1) / blockSize);
    dim3 grid_layer( (nlayer * nwno        + blockSize - 1) / blockSize);
    // dim3 grid_wn(    (nwno                 + blockSize - 1) / blockSize);
    dim3 grid_wn((nwno + blockSize - 1) / blockSize);
    // 1) Blackbody
    calculate_blackbody<<<grid_all, blockSize>>>(
        ctx.lvl_T_dev,
        ctx.wno_dev,
        ctx.wno0_dev,
        ctx.all_b_dev,
        nlevel,
        nwno,
        calc_type
    );

    double mu1 = 0.5;

    // 2) Initialize parameters (b0,b1, matrix stuff, boundaries)
    initialize_parameters<<<grid_layer, blockSize>>>(
        ctx.all_b_dev, ctx.b0_dev, ctx.b1_dev, ctx.dtau_og_dev,
        ctx.w0_no_raman_dev, ctx.cosb_og_dev, ctx.alpha_dev,
        ctx.lamda_dev, ctx.gama_dev, ctx.g1_plus_g2_dev,
        ctx.c_plus_up_dev, ctx.c_minus_up_dev,
        ctx.c_plus_down_dev, ctx.c_minus_down_dev,
        ctx.lvl_T_dev, ctx.lvl_P_dev,
        ctx.exptrm_dev, ctx.exptrm_positive_dev, ctx.exptrm_minus_dev,
        ctx.tau_top_dev, ctx.b_top_dev, ctx.b_surface_dev,ctx.surf_reflect_dev,
        mu1, hard_surface, nlevel, nwno
    );

    // 3) e-matrix
    calculate_e_matrix<<<grid_layer, blockSize>>>(
        ctx.exptrm_positive_dev, ctx.exptrm_minus_dev,
        ctx.gama_dev,
        ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
        nlayer * nwno
    );

    // 4) Tri-diagonal setup
    setup_tri_diag_all<<<grid_layer, blockSize>>>(
        nlayer, nwno,
        ctx.c_plus_up_dev, ctx.c_minus_up_dev,
        ctx.c_plus_down_dev, ctx.c_minus_down_dev,
        ctx.b_top_dev, ctx.b_surface_dev, ctx.surf_reflect_dev,
        ctx.gama_dev, ctx.dtau_og_dev,
        ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
        ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev
    );

    setup_tri_diag_1st<<<grid_wn, blockSize>>>(
        nlayer, nwno,
        ctx.c_plus_up_dev, ctx.c_minus_up_dev,
        ctx.c_plus_down_dev, ctx.c_minus_down_dev,
        ctx.b_top_dev, ctx.b_surface_dev, ctx.surf_reflect_dev,
        ctx.gama_dev, ctx.dtau_og_dev,
        ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
        ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev
    );

    setup_tri_diag_last<<<grid_wn, blockSize>>>(
        nlayer, nwno,
        ctx.c_plus_up_dev, ctx.c_minus_up_dev,
        ctx.c_plus_down_dev, ctx.c_minus_down_dev,
        ctx.b_top_dev, ctx.b_surface_dev, ctx.surf_reflect_dev,
        ctx.gama_dev, ctx.dtau_og_dev,
        ctx.e1_dev, ctx.e2_dev, ctx.e3_dev, ctx.e4_dev,
        ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev
    );

    // 5) Solve tri-diagonal
    // initialize_and_solve_tridiagonal<<<grid_layer, blockSize>>>(
    //     ctx.AS_dev, ctx.DS_dev, ctx.XK_dev,
    //     ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev,
    //     nlayer, nwno
    // );

    initialize_and_solve_tridiagonal<<<grid_wn, blockSize>>>(
        ctx.AS_dev, ctx.DS_dev, ctx.XK_dev,
        ctx.A_odd_dev, ctx.B_odd_dev, ctx.C_odd_dev, ctx.D_odd_dev,
        nlayer,  // this is ctx.nlayer = nlevel-1
        nwno
    );

    // 6) Matrices and exponents
    calculate_matrices_and_exponents<<<grid_layer, blockSize>>>(
        ctx.XK_dev, ctx.positive_dev, ctx.negative_dev,
        ctx.lamda_dev, ctx.gama_dev, ctx.b0_dev, ctx.b1_dev,
        ctx.g1_plus_g2_dev, mu1,
        ctx.G_matrix_dev, ctx.H_matrix_dev,
        ctx.J_matrix_dev, ctx.K_matrix_dev,
        ctx.alpha1_matrix_dev, ctx.alpha2_matrix_dev,
        ctx.sigma1_matrix_dev, ctx.sigma2_matrix_dev,
        ctx.exptrm_dev,
        ctx.exptrm_positive_mdpt_dev,
        ctx.exptrm_minus_mdpt_dev,
        nlevel, nwno
    );

    // 7) Loop over (g, t) combinations
    int block_flux = 256;
    int grid_flux  = (nwno + block_flux - 1) / block_flux;

    for (int i_iter = 0; i_iter < ng * nt; ++i_iter) {
        double ubar1_i = ubar1[i_iter];

        // boundary fluxes
        fill_flux_plus_minus<<<grid_wn, blockSize>>>(
            ctx.all_b_dev, ctx.b1_dev,
            ctx.tau_top_dev, ctx.surf_reflect_dev, ubar1_i,
            ctx.flux_plus_dev, ctx.flux_minus_dev,
            hard_surface, nlevel, nwno
        );

        // angle exponentials
        fill_exptrm_angle<<<grid_layer, blockSize>>>(
            ctx.dtau_og_dev, ctx.exptrm_angle_dev, ctx.exptrm_angle_mdpt_dev,
            ubar1_i, nlevel, nwno
        );

        // Combined sweep kernel (TOP optimization: one launch per i_iter)
        update_flux_combined_one_kernel_loop_new<<<grid_flux, block_flux>>>(
            ctx.flux_minus_dev, ctx.flux_minus_mdpt_dev,
            ctx.flux_plus_dev,  ctx.flux_plus_mdpt_dev,
            ctx.exptrm_angle_dev,       ctx.exptrm_angle_mdpt_dev,
            ctx.J_matrix_dev,           ctx.G_matrix_dev,
            ctx.lamda_dev,
            ctx.exptrm_positive_dev,    ctx.exptrm_positive_mdpt_dev,
            ctx.K_matrix_dev,           ctx.H_matrix_dev,
            ctx.exptrm_minus_dev,       ctx.exptrm_minus_mdpt_dev,
            ctx.sigma1_matrix_dev,      ctx.sigma2_matrix_dev,
            ctx.alpha1_matrix_dev,      ctx.alpha2_matrix_dev,
            ctx.dtau_og_dev,
            ubar1_i,
            nlevel, nwno
        );

        // Save all level fluxes
        dim3 block_save(16,16);
        dim3 grid_save((nlevel * nwno + block_save.x * block_save.y - 1)
                       / (block_save.x * block_save.y));

        get_lvl_fluxes<<<grid_save, block_save>>>(
            ctx.flux_minus_dev, ctx.flux_plus_dev,
            ctx.flux_minus_mdpt_dev, ctx.flux_plus_mdpt_dev,
            ctx.flux_minus_all_dev, ctx.flux_plus_all_dev,
            ctx.flux_minus_mdpt_all_dev, ctx.flux_plus_mdpt_all_dev,
            i_iter, nlevel, nwno
        );

        // Save top flux
        get_fluxes_at_top<<<grid_save, block_save>>>(
            ctx.flux_at_top_dev,
            ctx.flux_plus_mdpt_dev,
            i_iter, nlevel, nwno
        );
    }


}
extern "C" void get_thermal_1d_free()
{
    ThermalContext &ctx = g_ctx;
    if (!ctx.initialized) return;

    auto FREE = [](double *&p) {
        if (p) { CUDA_CHECK(cudaFree(p)); p = nullptr; }
    };


    FREE(ctx.alpha_dev);
    FREE(ctx.lamda_dev);
    FREE(ctx.gama_dev);
    FREE(ctx.c_plus_up_dev);
    FREE(ctx.c_minus_up_dev);
    FREE(ctx.c_plus_down_dev);
    FREE(ctx.c_minus_down_dev);
    FREE(ctx.b_surface_dev);
    FREE(ctx.surf_reflect_dev);
    FREE(ctx.exptrm_positive_dev);
    FREE(ctx.exptrm_minus_dev);
    FREE(ctx.exptrm_dev);

    FREE(ctx.all_b_dev);
    FREE(ctx.b0_dev);
    FREE(ctx.b1_dev);
    FREE(ctx.g1_plus_g2_dev);

    FREE(ctx.tau_top_dev);
    FREE(ctx.b_top_dev);

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
    FREE(ctx.J_matrix_dev);
    FREE(ctx.K_matrix_dev);
    FREE(ctx.alpha1_matrix_dev);
    FREE(ctx.alpha2_matrix_dev);
    FREE(ctx.sigma1_matrix_dev);
    FREE(ctx.sigma2_matrix_dev);

    FREE(ctx.exptrm_positive_mdpt_dev);
    FREE(ctx.exptrm_minus_mdpt_dev);

    FREE(ctx.flux_minus_dev);
    FREE(ctx.flux_plus_dev);
    FREE(ctx.flux_minus_mdpt_dev);
    FREE(ctx.flux_plus_mdpt_dev);


    FREE(ctx.exptrm_angle_dev);
    FREE(ctx.exptrm_angle_mdpt_dev);

    ctx.initialized = false;
}
