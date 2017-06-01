#ifndef GEMM_H
#define GEMM_H

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
#include "opencl.h"
cl_kernel get_gemm_kernel();
cl_kernel get_gemm_nn_kernel();
cl_kernel get_gemm_nt_kernel();
cl_kernel get_gemm_tn_kernel();
cl_kernel get_gemm_nn_fast_kernel();


void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        cl_mem A_gpu, int lda, 
        cl_mem B_gpu, int ldb,
        float BETA,
        cl_mem C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float * A, int lda, 
        float * B, int ldb,
        float BETA,
        float * C, int ldc);

void gemm_ongpu_fast(int TA, int TB, int M, int N, int K, float ALPHA, 
        cl_mem A_gpu, int lda, 
        cl_mem B_gpu, int ldb,
        float BETA,
        cl_mem C_gpu, int ldc);

#endif
#endif
