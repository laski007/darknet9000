#include "blas.h"
#include "math.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff < 0) ? 1 : -1;
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1 : -1;
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

#ifdef GPU
#include "opencl.h"
#include <assert.h>
#define BLOCK 64
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
cl_kernel get_scale_bias_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "scale_bias_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void scale_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size)
{
    //cl_mem output = (cl_mem) cuda_output;
    //cl_mem biases = (cl_mem) cuda_biases;
    cl_kernel kernel = get_scale_bias_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(output), (void*) &output);
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases), (void*) &biases);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl_check_error(cl);

    const size_t gsize[] = {size, n, batch};
    const size_t localws[] = {BLOCK, 1, 1};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 3, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_backward_scale_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "backward_scale_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void backward_scale_gpu(cl_mem x_norm, cl_mem delta, int batch, int n, int size, cl_mem scale_updates)
{
    //cl_mem x_norm = (cl_mem) cuda_x_norm;
    //cl_mem delta = (cl_mem) cuda_delta;
    //cl_mem scale_updates = (cl_mem) cuda_scale_updates;
    cl_kernel kernel = get_backward_scale_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x_norm), (void*) &x_norm);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(scale_updates), (void*) &scale_updates);
    cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_add_bias_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "add_bias_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void add_bias_gpu(cl_mem output, cl_mem biases, int batch, int n, int size)
{
    //cl_mem output = (cl_mem) cuda_output;
    //cl_mem biases = (cl_mem) cuda_biases;
    cl_kernel kernel = get_add_bias_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(output), (void*) &output);
    cl.error = clSetKernelArg(kernel, i++, sizeof(biases), (void*) &biases);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl_check_error(cl);

    const size_t gsize[] = {size, n, batch};
    const size_t localws[] = {BLOCK, 1, 1};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 3, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_backward_bias_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "backward_bias_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void backward_bias_gpu(cl_mem bias_updates, cl_mem delta, int batch, int n, int size)
{
    //cl_mem bias_updates = (cl_mem) cuda_bias_updates;
    //cl_mem delta = (cl_mem) cuda_delta;
    cl_kernel kernel = get_backward_bias_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(bias_updates), (void*) &bias_updates);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_adam_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "adam_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void adam_gpu(int n, cl_mem x, cl_mem m, cl_mem v, float B1, float B2, float rate, float eps, int t)
{
    //cl_mem x = (cl_mem) cuda_x;
    //cl_mem m = (cl_mem) cuda_m;
    //cl_mem v = (cl_mem) cuda_v;
    cl_kernel kernel = get_adam_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(m), (void*) &m);
    cl.error = clSetKernelArg(kernel, i++, sizeof(v), (void*) &v);
    cl.error = clSetKernelArg(kernel, i++, sizeof(B1), (void*) &B1);
	cl.error = clSetKernelArg(kernel, i++, sizeof(B2), (void*) &B2);
	cl.error = clSetKernelArg(kernel, i++, sizeof(rate), (void*) &rate);
	cl.error = clSetKernelArg(kernel, i++, sizeof(eps), (void*) &eps);
	cl.error = clSetKernelArg(kernel, i++, sizeof(t), (void*) &t);
    cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_normalize_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "normalize_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void normalize_gpu(cl_mem x, cl_mem mean, cl_mem variance, int batch, int filters, int spatial)
{
    //cl_mem x = (cl_mem) cuda_x;
    //cl_mem mean = (cl_mem) cuda_mean;
    //cl_mem variance = (cl_mem) cuda_variance;
    int N = batch*filters*spatial;
    cl_kernel kernel = get_normalize_kernel();
    cl_command_queue queue = cl.queue;


    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*) &mean);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*) &variance);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
	cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
	cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_normalize_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "normalize_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void normalize_delta_gpu(cl_mem x, cl_mem mean, cl_mem variance, cl_mem mean_delta, cl_mem variance_delta, int batch, int filters, int spatial, cl_mem delta)
{
    //cl_mem x = (cl_mem) cuda_x;
    //cl_mem mean = (cl_mem) cuda_mean;
    //cl_mem variance = (cl_mem) cuda_variance;
	//cl_mem mean_delta = (cl_mem) cuda_mean_delta;
    //cl_mem variance_delta = (cl_mem) cuda_variance_delta;
    //cl_mem delta = (cl_mem) cuda_delta;

    cl_kernel kernel = get_normalize_delta_kernel();
    cl_command_queue queue = cl.queue;
	size_t N = batch*filters*spatial;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*) &mean);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*) &variance);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta), (void*) &mean_delta);
	cl.error = clSetKernelArg(kernel, i++, sizeof(variance_delta), (void*) &variance_delta);
	cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
	cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
	cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
	cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_fast_mean_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "fast_mean_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_mean_delta_gpu(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta)
{
    //cl_mem delta = (cl_mem) cuda_delta;
    //cl_mem variance = (cl_mem) cuda_variance;
	//cl_mem mean_delta = (cl_mem) cuda_mean_delta;

    cl_kernel kernel = get_fast_mean_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*) &variance);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
	cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta), (void*) &mean_delta);

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_fast_variance_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "fast_variance_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_variance_delta_gpu(cl_mem x, cl_mem delta, cl_mem mean, cl_mem variance, int batch, int filters, int spatial, cl_mem variance_delta)
{
    //cl_mem x = (cl_mem) cuda_x;
    //cl_mem delta = (cl_mem) cuda_delta;
	//cl_mem mean = (cl_mem) cuda_mean;
	//cl_mem variance = (cl_mem) cuda_variance;
	//cl_mem variance_delta = (cl_mem) cuda_variance_delta;

    cl_kernel kernel = get_fast_variance_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*) &mean);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*) &variance);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
	cl.error = clSetKernelArg(kernel, i++, sizeof(variance_delta), (void*) &variance_delta);

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_mean_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "mean_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mean_delta_gpu(cl_mem delta, cl_mem variance, int batch, int filters, int spatial, cl_mem mean_delta)
{
    //cl_mem delta = (cl_mem) cuda_delta;
	//cl_mem variance = (cl_mem) cuda_variance;
	//cl_mem mean_delta = (cl_mem) cuda_mean_delta;

    cl_kernel kernel = get_mean_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*) &variance);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
	cl.error = clSetKernelArg(kernel, i++, sizeof(mean_delta), (void*) &mean_delta);

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_mean_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "mean_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mean_gpu(cl_mem x, int batch, int filters, int spatial, cl_mem mean)
{
    //cl_mem x = (cl_mem) cuda_x;
	//cl_mem mean = (cl_mem) cuda_mean;

    cl_kernel kernel = get_mean_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
	cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*) &mean);

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_variance_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "variance_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void variance_gpu(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance)
{
    //cl_mem x = (cl_mem) cuda_x;
	//cl_mem mean = (cl_mem) cuda_mean;
	//cl_mem variance = (cl_mem) cuda_variance;
    cl_kernel kernel = get_variance_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*) &mean);
	cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
	cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*) &variance);

    cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_reorg_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "reorg_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void reorg_ongpu(cl_mem x, int w, int h, int c, int batch, int stride, int forward, cl_mem out)
{
    //cl_mem x = (cl_mem) cuda_x;
	//cl_mem out = (cl_mem) cuda_out;

    cl_kernel kernel = get_reorg_kernel();
    cl_command_queue queue = cl.queue;
	int size = w*h*c*batch;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
	cl.error = clSetKernelArg(kernel, i++, sizeof(w), (void*) &w);
    cl.error = clSetKernelArg(kernel, i++, sizeof(h), (void*) &h);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c), (void*) &c);
	cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
	cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
	cl.error = clSetKernelArg(kernel, i++, sizeof(forward), (void*) &forward);
	cl.error = clSetKernelArg(kernel, i++, sizeof(out), (void*) &out);
    cl_check_error(cl);

    const size_t gsize[] = {size*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_axpy_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "axpy_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void axpy_ongpu_offset(int N, float ALPHA, cl_mem X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY)
{
    //cl_mem X = (cl_mem) cuda_X;
	//cl_mem Y = (cl_mem) cuda_Y;

    cl_kernel kernel = get_axpy_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(OFFX), (void*) &OFFX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
	cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*) &Y);
	cl.error = clSetKernelArg(kernel, i++, sizeof(OFFY), (void*) &OFFY);
	cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);
	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

void axpy_ongpu(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY)
{
    axpy_ongpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

cl_kernel get_pow_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "pow_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void pow_ongpu(int N, float ALPHA, cl_mem X, int INCX, cl_mem Y, int INCY)
{
    //cl_mem X = (cl_mem) cuda_X;
	//cl_mem Y = (cl_mem) cuda_Y;

    cl_kernel kernel = get_pow_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
	cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*) &Y);
	cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);
	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_const_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "const_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void const_ongpu(int N, float ALPHA, cl_mem X, int INCX)
{
    //cl_mem X = (cl_mem) cuda_X;

    cl_kernel kernel = get_const_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_constrain_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "constrain_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void constrain_ongpu(int N, float ALPHA, cl_mem X, int INCX)
{
    //cl_mem X = (cl_mem) cuda_X;

    cl_kernel kernel = get_constrain_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_supp_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "supp_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void supp_ongpu(int N, float ALPHA, cl_mem X, int INCX)
{
    //cl_mem X = (cl_mem) cuda_X;

    cl_kernel kernel = get_supp_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_add_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "add_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void add_ongpu(int N, float ALPHA, cl_mem X, int INCX)
{
    //cl_mem X = (cl_mem) cuda_X;

    cl_kernel kernel = get_add_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_scal_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "scal_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void scal_ongpu(int N, float ALPHA, cl_mem X, int INCX)
{
    //cl_mem X = (cl_mem) cuda_X;

    cl_kernel kernel = get_scal_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_fill_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "fill_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fill_ongpu(int N, float ALPHA, cl_mem X, int INCX)
{
    //cl_mem X = (cl_mem) cuda_X;

    cl_kernel kernel = get_fill_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ALPHA), (void*) &ALPHA);
	cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);

	cl_check_error(cl);

    const size_t gsize[] = {N/*BLOCK*/};
    //const size_t localws[] = ;//{};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, NULL, 0, 0, 0);
    if(cl.error == -5) cl.error = 0;
    cl_check_error(cl);
}

cl_kernel get_mask_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "mask_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mask_ongpu(int N, cl_mem X, float mask_num, cl_mem mask)
{
    //cl_mem X = (cl_mem) cuda_X;
	//cl_mem mask = (cl_mem) cuda_mask;
    cl_kernel kernel = get_mask_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
	cl.error = clSetKernelArg(kernel, i++, sizeof(mask_num), (void*) &mask_num);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask), (void*) &mask);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_copy_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "copy_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void copy_ongpu_offset(int N, cl_mem X, int OFFX, int INCX, cl_mem Y, int OFFY, int INCY)
{
    //cl_mem X = (cl_mem) cuda_X;
	//cl_mem Y = (cl_mem) cuda_Y;
    cl_kernel kernel = get_copy_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
	cl.error = clSetKernelArg(kernel, i++, sizeof(OFFX), (void*) &OFFX);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
	cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*) &Y);
	cl.error = clSetKernelArg(kernel, i++, sizeof(OFFY), (void*) &OFFY);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

void copy_ongpu(int N, cl_mem X, int INCX, cl_mem Y, int INCY)
{
    copy_ongpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

cl_kernel get_mul_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "mul_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mul_ongpu(int N, cl_mem X, int INCX, cl_mem Y, int INCY)
{
    //cl_mem X = (cl_mem) cuda_X;
	//cl_mem Y = (cl_mem) cuda_Y;
    cl_kernel kernel = get_mul_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(N), (void*) &N);
    cl.error = clSetKernelArg(kernel, i++, sizeof(X), (void*) &X);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCX), (void*) &INCX);
	cl.error = clSetKernelArg(kernel, i++, sizeof(Y), (void*) &Y);
    cl.error = clSetKernelArg(kernel, i++, sizeof(INCY), (void*) &INCY);

	cl_check_error(cl);

    const size_t gsize[] = {N*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_fast_mean_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "fast_mean_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_mean_gpu(cl_mem x, int batch, int filters, int spatial, cl_mem mean)
{
    //cl_mem x = (cl_mem) cuda_x;
	//cl_mem mean = (cl_mem) cuda_mean;
    cl_kernel kernel = get_fast_mean_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
	cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*) &mean);

	cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_fast_variance_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "fast_variance_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void fast_variance_gpu(cl_mem x, cl_mem mean, int batch, int filters, int spatial, cl_mem variance)
{
    //cl_mem x = (cl_mem) cuda_x;
	//cl_mem mean = (cl_mem) cuda_mean;
	//cl_mem variance = (cl_mem) cuda_variance;
    cl_kernel kernel = get_fast_variance_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
	cl.error = clSetKernelArg(kernel, i++, sizeof(mean), (void*) &mean);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(filters), (void*) &filters);
	cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(variance), (void*) &variance);

	cl_check_error(cl);

    const size_t gsize[] = {filters*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_flatten_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "flatten_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void flatten_ongpu(cl_mem x, int spatial, int layers, int batch, int forward, cl_mem out)
{
    //cl_mem x = (cl_mem) cuda_x;
	//cl_mem out = (cl_mem) cuda_out;

    cl_kernel kernel = get_flatten_kernel();
    cl_command_queue queue = cl.queue;
	int size = spatial*batch*layers;
    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
	cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(spatial), (void*) &spatial);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layers), (void*) &layers);
	cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
	cl.error = clSetKernelArg(kernel, i++, sizeof(forward), (void*) &forward);
    cl.error = clSetKernelArg(kernel, i++, sizeof(out), (void*) &out);

	cl_check_error(cl);

    const size_t gsize[] = {size*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_shortcut_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "shortcut_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem add, int w2, int h2, int c2, cl_mem out)
{
   // cl_mem add = (cl_mem) cuda_add;
	//cl_mem out = (cl_mem) cuda_out;

    cl_kernel kernel = get_shortcut_kernel();
    cl_command_queue queue = cl.queue;

	int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
	cl.error = clSetKernelArg(kernel, i++, sizeof(minw), (void*) &minw);
    cl.error = clSetKernelArg(kernel, i++, sizeof(minh), (void*) &minh);
    cl.error = clSetKernelArg(kernel, i++, sizeof(minc), (void*) &minc);
	cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
	cl.error = clSetKernelArg(kernel, i++, sizeof(sample), (void*) &sample);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
	cl.error = clSetKernelArg(kernel, i++, sizeof(w1), (void*) &w1);
	cl.error = clSetKernelArg(kernel, i++, sizeof(h1), (void*) &h1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c1), (void*) &c1);
    cl.error = clSetKernelArg(kernel, i++, sizeof(add), (void*) &add);
	cl.error = clSetKernelArg(kernel, i++, sizeof(w2), (void*) &w2);
	cl.error = clSetKernelArg(kernel, i++, sizeof(h2), (void*) &h2);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c2), (void*) &c2);
	cl.error = clSetKernelArg(kernel, i++, sizeof(out), (void*) &out);
	cl_check_error(cl);

    const size_t gsize[] = {size*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_smooth_l1_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "smooth_l1_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void smooth_l1_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error)
{

    cl_kernel kernel = get_smooth_l1_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
	cl.error = clSetKernelArg(kernel, i++, sizeof(pred), (void*) &pred);
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth), (void*) &truth);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
	cl.error = clSetKernelArg(kernel, i++, sizeof(error), (void*) &error);

	cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_l2_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "l2_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void l2_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error)
{
    //cl_mem pred = (cl_mem) cuda_pred;
	//cl_mem truth = (cl_mem) cuda_truth;
	//cl_mem delta = (cl_mem) cuda_delta;
	//cl_mem error = (cl_mem) cuda_error;
    cl_kernel kernel = get_l2_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
	cl.error = clSetKernelArg(kernel, i++, sizeof(pred), (void*) &pred);
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth), (void*) &truth);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
	cl.error = clSetKernelArg(kernel, i++, sizeof(error), (void*) &error);

	cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_l1_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "l1_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void l1_gpu(int n, cl_mem pred, cl_mem truth, cl_mem delta, cl_mem error)
{
    //cl_mem pred = (cl_mem) cuda_pred;
	//cl_mem truth = (cl_mem) cuda_truth;
	//cl_mem delta = (cl_mem) cuda_delta;
	//cl_mem error = (cl_mem) cuda_error;
    cl_kernel kernel = get_l1_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
	cl.error = clSetKernelArg(kernel, i++, sizeof(pred), (void*) &pred);
    cl.error = clSetKernelArg(kernel, i++, sizeof(truth), (void*) &truth);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
	cl.error = clSetKernelArg(kernel, i++, sizeof(error), (void*) &error);

	cl_check_error(cl);

    const size_t gsize[] = {n*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_weighted_sum_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "weighted_sum_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void weighted_sum_gpu(cl_mem a, cl_mem b, cl_mem s, int num, cl_mem c)
{
    //cl_mem a = (cl_mem) cuda_a;
	//cl_mem b = (cl_mem) cuda_b;
	//cl_mem s = (cl_mem) cuda_s;
	//cl_mem c = (cl_mem) cuda_c;
    cl_kernel kernel = get_weighted_sum_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(num), (void*) &num);
	cl.error = clSetKernelArg(kernel, i++, sizeof(a), (void*) &a);
    cl.error = clSetKernelArg(kernel, i++, sizeof(b), (void*) &b);
    cl.error = clSetKernelArg(kernel, i++, sizeof(s), (void*) &s);
	cl.error = clSetKernelArg(kernel, i++, sizeof(c), (void*) &c);

	cl_check_error(cl);

    const size_t gsize[] = {num*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_weighted_delta_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "weighted_delta_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void weighted_delta_gpu(cl_mem a, cl_mem b, cl_mem s, cl_mem da, cl_mem db, cl_mem ds, int num, cl_mem dc)
{
    //cl_mem a = (cl_mem) cuda_a;
	//cl_mem b = (cl_mem) cuda_b;
	//cl_mem s = (cl_mem) cuda_s;
    //cl_mem da = (cl_mem) cuda_da;
	//cl_mem db = (cl_mem) cuda_db;
	//cl_mem ds = (cl_mem) cuda_ds;
	//cl_mem dc = (cl_mem) cuda_dc;
    cl_kernel kernel = get_weighted_delta_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(num), (void*) &num);
	cl.error = clSetKernelArg(kernel, i++, sizeof(a), (void*) &a);
    cl.error = clSetKernelArg(kernel, i++, sizeof(b), (void*) &b);
    cl.error = clSetKernelArg(kernel, i++, sizeof(s), (void*) &s);
	cl.error = clSetKernelArg(kernel, i++, sizeof(da), (void*) &da);
	cl.error = clSetKernelArg(kernel, i++, sizeof(db), (void*) &db);
	cl.error = clSetKernelArg(kernel, i++, sizeof(ds), (void*) &ds);
	cl.error = clSetKernelArg(kernel, i++, sizeof(dc), (void*) &dc);
	cl_check_error(cl);

    const size_t gsize[] = {num*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

cl_kernel get_mult_add_into_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "mult_add_into_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void mult_add_into_gpu(int num, cl_mem a, cl_mem b, cl_mem c)
{
    //cl_mem a = (cl_mem) cuda_a;
	//cl_mem b = (cl_mem) cuda_b;
	//cl_mem c = (cl_mem) cuda_c;

    cl_kernel kernel = get_mult_add_into_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(num), (void*) &num);
	cl.error = clSetKernelArg(kernel, i++, sizeof(a), (void*) &a);
    cl.error = clSetKernelArg(kernel, i++, sizeof(b), (void*) &b);
    cl.error = clSetKernelArg(kernel, i++, sizeof(c), (void*) &c);

	cl_check_error(cl);

    const size_t gsize[] = {num*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}


cl_kernel get_softmax_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/blas_kernels.cl", "softmax_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void softmax_gpu(cl_mem input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem output)
{
    //cl_mem input = (cl_mem) cuda_input;
	//cl_mem output = (cl_mem) cuda_output;

    cl_kernel kernel = get_softmax_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(input), (void*) &input);
	cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch), (void*) &batch);
    cl.error = clSetKernelArg(kernel, i++, sizeof(batch_offset), (void*) &batch_offset);
    cl.error = clSetKernelArg(kernel, i++, sizeof(groups), (void*) &groups);
	cl.error = clSetKernelArg(kernel, i++, sizeof(group_offset), (void*) &group_offset);
    cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(temp), (void*) &temp);
    cl.error = clSetKernelArg(kernel, i++, sizeof(output), (void*) &output);

	cl_check_error(cl);

    const size_t gsize[] = {batch*groups*BLOCK};
    const size_t localws[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}






#endif
