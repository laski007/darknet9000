#include "dropout_layer.h"
#include "utils.h"
//#include "cuda.h"
#include "opencl.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cl_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    //cuda_free(l->rand_gpu);
	clReleaseMemObject(l->rand_gpu);

    l->rand_gpu = cl_make_array(l->rand, inputs*l->batch);
    #endif
}

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}

#ifdef GPU

#define BLOCK 64
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

cl_kernel get_yoloswag420blazeit360noscope()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/dropout_layer_kernels.cl", "yoloswag420blazeit360noscope", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    cl_kernel kernel = get_yoloswag420blazeit360noscope();
    cl_command_queue queue = cl.queue;

    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cl_random(layer.rand_gpu, size);

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(net.input_gpu), (void*) &(net.input_gpu));
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_gpu), (void*) &(layer.rand_gpu));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.probability), (void*) &(layer.probability));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.scale), (void*) &(layer.scale));
    cl_check_error(cl);

    const size_t gsize[] = {size*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    cl_kernel kernel = get_yoloswag420blazeit360noscope();
    cl_command_queue queue = cl.queue;

    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(net.delta_gpu), (void*) &(net.delta_gpu));
    cl.error = clSetKernelArg(kernel, i++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_gpu), (void*) &(layer.rand_gpu));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.probability), (void*) &(layer.probability));
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.scale), (void*) &(layer.scale));

    cl_check_error(cl);

    const size_t gsize[] = {size*BLOCK};
    const size_t localws[] = {BLOCK};
    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, gsize, localws, 0, 0, 0);
    cl_check_error(cl);
}

#endif




