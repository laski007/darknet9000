#ifndef BLAS_H
#define BLAS_H
void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void const_cpu(int N, float ALPHA, float *X, int INCX);

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);

#ifdef GPU
//#include "cuda.h"
#include "opencl.h"
cl_kernel get_scale_bias_kernel();
void scale_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size);

cl_kernel get_backward_scale_kernel();
void backward_scale_gpu(cl_mem x_norm, cl_mem delta, int batch, int n, int size, cl_mem scale_updates);

cl_kernel get_add_bias_kernel();

void add_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size);

cl_kernel get_backward_bias_kernel();
void backward_bias_gpu(cl_mem bias_updates, cl_mem delta, int batch, int n, int size);

cl_kernel get_adam_kernel();
void adam_gpu(int n, cl_mem x, cl_mem m, cl_mem v, float B1, float B2, float rate, float eps, int t);

cl_kernel get_normalize_kernel();
void normalize_gpu(cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial);

cl_kernel get_normalize_delta_kernel();
void normalize_delta_gpu(cl_mem x, cl_mem mean, cl_mem variance, cl_mem mean_delta, cl_mem variance_delta, int batch, int filters, int spatial, cl_mem delta);

cl_kernel get_fast_mean_delta_kernel();
void fast_mean_delta_gpu(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta);

cl_kernel get_fast_variance_delta_kernel();
void fast_variance_delta_gpu(cl_mem x, cl_mem delta, cl_mem mean, cl_mem variance, int batch, int filters, int spatial, cl_mem variance_delta);

cl_kernel get_mean_delta_kernel();
void mean_delta_gpu(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta);

cl_kernel get_mean_kernel();
void mean_gpu(cl_mem x, int batch, int filters, int spatial, cl_mem mean);

cl_kernel get_variance_kernel();
void variance_gpu(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance);

cl_kernel get_reorg_kernel();
void reorg_ongpu(cl_mem x, int w, int h, int c, int batch, int stride, int forward, cl_mem out);

cl_kernel get_axpy_kernel();
void axpy_ongpu_offset(int N, float ALPHA, cl_mem X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY);
void axpy_ongpu(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY);

cl_kernel get_pow_kernel();
void pow_ongpu(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY);

cl_kernel get_const_kernel();
void const_ongpu(int N, float ALPHA, cl_mem X, int INCX);

cl_kernel get_constrain_kernel();
void constrain_ongpu(int N, float ALPHA, cl_mem X, int INCX);

cl_kernel get_supp_kernel();
void supp_ongpu(int N, float ALPHA, cl_mem X, int INCX);

cl_kernel get_add_kernel();
void add_ongpu(int N, float ALPHA, cl_mem X, int INCX);

cl_kernel get_scal_kernel();
void scal_ongpu(int N, float ALPHA, cl_mem X, int INCX);

cl_kernel get_fill_kernel();
void fill_ongpu(int N, float ALPHA, cl_mem X, int INCX);

cl_kernel get_mask_kernel();
void mask_ongpu(int N, cl_mem X, float mask_num, cl_mem mask);

cl_kernel get_copy_kernel();
void copy_ongpu_offset(int N, cl_mem X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY);
void copy_ongpu(int N, cl_mem X, int INCX, cl_mem Y, int INCY);

cl_kernel get_mul_kernel();
void mul_ongpu(int N, cl_mem X, int INCX, cl_mem Y, int INCY);

cl_kernel get_fast_mean_kernel();
void fast_mean_gpu(cl_mem x, int batch, int filters, int spatial, cl_mem mean);

cl_kernel get_fast_variance_kernel();
void fast_variance_gpu(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance);

cl_kernel get_flatten_kernel();
void flatten_ongpu(cl_mem x, int spatial, int layers, int batch, int forward, cl_mem out);

cl_kernel get_shortcut_kernel();
void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, cl_mem out);

cl_kernel get_smooth_l1_kernel();
void smooth_l1_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error);

cl_kernel get_l2_kernel();
void l2_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error);

cl_kernel get_l1_kernel();
void l1_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error);

cl_kernel get_weighted_sum_kernel();
void weighted_sum_gpu(cl_mem a, cl_mem b, cl_mem s, int num, cl_mem c);

cl_kernel get_weighted_delta_kernel();
void weighted_delta_gpu(cl_mem a, cl_mem b, cl_mem s, cl_mem da, cl_mem db, cl_mem ds, int num, cl_mem dc);

cl_kernel get_mult_add_into_kernel();
void mult_add_into_gpu(int num, cl_mem a, cl_mem b, cl_mem c);

cl_kernel get_softmax_kernel();
void softmax_gpu(cl_mem input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output);

#endif
#endif
