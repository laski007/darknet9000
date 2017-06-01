#include "avgpool_layer.h"
//#include "cuda.h"
#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cl_make_array(l.output, output_size);
    l.delta_gpu   = cl_make_array(l.delta, output_size);
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}

#ifdef GPU
#include "opencl.h"
#define BLOCK 64
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

cl_kernel get_avgpool_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/avgpool_layer_kernels.cl", "forward_avgpool_layer_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;

}
void forward_avgpool_layer_gpu(avgpool_layer l, network net)
{
    cl_kernel kernel = get_avgpool_kernel();
    cl_command_queue queue = cl.queue;

    size_t n = l.c*l.batch;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(l.w), (void*) &(l.w));
    cl.error = clSetKernelArg(kernel, i++, sizeof(l.h), (void*) &(l.h));
	cl.error = clSetKernelArg(kernel, i++, sizeof(l.c), (void*) &(l.c));
    cl.error = clSetKernelArg(kernel, i++, sizeof(net.input_gpu), (void*) &(net.input_gpu));
    cl.error = clSetKernelArg(kernel, i++, sizeof(l.output_gpu), (void*) &(l.output_gpu));
    cl_check_error(cl);

    size_t gsize[] = {n*BLOCK};
    size_t localws[]={BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);

	
}

cl_kernel get_backward_avgpool_layer_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/avgpool_layer_kernels.cl", "backward_avgpool_layer_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;

}

void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    cl_kernel kernel = get_backward_avgpool_layer_kernel();
    cl_command_queue queue = cl.queue;

    size_t n = layer.c*layer.batch;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.w), (void*) &(layer.w));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.h), (void*) &(layer.h));
	cl.error = clSetKernelArg(kernel, i++, sizeof(layer.c), (void*) &(layer.c));
    cl.error = clSetKernelArg(kernel, i++, sizeof(net.delta_gpu), (void*) &(net.delta_gpu));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.delta_gpu), (void*) &(layer.delta_gpu));
    cl_check_error(cl);

    size_t gsize[] = {n*BLOCK};
    size_t localws[]={BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}



#endif





























