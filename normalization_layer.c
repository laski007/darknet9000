#include "normalization_layer.h"
#include "blas.h"
#include "opencl.h"
#include <stdio.h>

layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
    fprintf(stderr, "Local Response Normalization Layer: %d x %d x %d image, %d size\n", w,h,c,size);
    layer layer = {0};
    layer.type = NORMALIZATION;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.kappa = kappa;
    layer.size = size;
    layer.alpha = alpha;
    layer.beta = beta;
    layer.output = calloc(h * w * c * batch, sizeof(float));
    layer.delta = calloc(h * w * c * batch, sizeof(float));
    layer.squared = calloc(h * w * c * batch, sizeof(float));
    layer.norms = calloc(h * w * c * batch, sizeof(float));
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    layer.forward = forward_normalization_layer;
    layer.backward = backward_normalization_layer;
    #ifdef GPU
    layer.forward_gpu = forward_normalization_layer_gpu;
    layer.backward_gpu = backward_normalization_layer_gpu;

    layer.output_gpu =  cl_make_array(layer.output, h * w * c * batch);
    layer.delta_gpu =   cl_make_array(layer.delta, h * w * c * batch);
    layer.squared_gpu = cl_make_array(layer.squared, h * w * c * batch);
    layer.norms_gpu =   cl_make_array(layer.norms, h * w * c * batch);
    #endif
    return layer;
}

void resize_normalization_layer(layer *layer, int w, int h)
{
    int c = layer->c;
    int batch = layer->batch;
    layer->h = h;
    layer->w = w;
    layer->out_h = h;
    layer->out_w = w;
    layer->inputs = w*h*c;
    layer->outputs = layer->inputs;
    layer->output = realloc(layer->output, h * w * c * batch * sizeof(float));
    layer->delta = realloc(layer->delta, h * w * c * batch * sizeof(float));
    layer->squared = realloc(layer->squared, h * w * c * batch * sizeof(float));
    layer->norms = realloc(layer->norms, h * w * c * batch * sizeof(float));
#ifdef GPU
    //cuda_free(layer->output_gpu);
    //cuda_free(layer->delta_gpu); 
    //cuda_free(layer->squared_gpu); 
    //cuda_free(layer->norms_gpu);
	clReleaseMemObject(layer->output_gpu);
	clReleaseMemObject(layer->delta_gpu);
	clReleaseMemObject(layer->squared_gpu);
	clReleaseMemObject(layer->norms_gpu);
   
    layer->output_gpu =  cl_make_array(layer->output, h * w * c * batch);
    layer->delta_gpu =   cl_make_array(layer->delta, h * w * c * batch);
    layer->squared_gpu = cl_make_array(layer->squared, h * w * c * batch);
    layer->norms_gpu =   cl_make_array(layer->norms, h * w * c * batch);
#endif
}

void forward_normalization_layer(const layer layer, network net)
{
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    scal_cpu(w*h*c*layer.batch, 0, layer.squared, 1);

    for(b = 0; b < layer.batch; ++b){
        float *squared = layer.squared + w*h*c*b;
        float *norms   = layer.norms + w*h*c*b;
        float *input   = net.input + w*h*c*b;
        pow_cpu(w*h*c, 2, input, 1, squared, 1);

        const_cpu(w*h, layer.kappa, norms, 1);
        for(k = 0; k < layer.size/2; ++k){
            axpy_cpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
        }

        for(k = 1; k < layer.c; ++k){
            copy_cpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            if(prev >= 0)      axpy_cpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            if(next < layer.c) axpy_cpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
    mul_cpu(w*h*c*layer.batch, net.input, 1, layer.output, 1);
}

void backward_normalization_layer(const layer layer, network net)
{
    // TODO This is approximate ;-)
    // Also this should add in to delta instead of overwritting.

    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    pow_cpu(w*h*c*layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
    mul_cpu(w*h*c*layer.batch, layer.delta, 1, net.delta, 1);
}

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net)
{
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    scal_ongpu(w*h*c*layer.batch, 0, layer.squared_gpu, 1);

    for(b = 0; b < layer.batch; ++b){
		float * new = (float *) layer.squared_gpu;
    	float *squared_float = new + w*h*c*b;
		cl_mem squared = (cl_mem) squared_float;

		float * norm_float = (float *) layer.norms_gpu;
    	float *norms_float   =  norm_float + w*h*c*b;
		cl_mem norms = (cl_mem) norms_float;

        float *input_float   = net.input_gpu + w*h*c*b;
		cl_mem input = (cl_mem) input_float;
        pow_ongpu(w*h*c, 2, input, 1, squared, 1);

        const_ongpu(w*h, layer.kappa, norms, 1);
        for(k = 0; k < layer.size/2; ++k){
			float * new = (float *) squared;
    		float *squared_float = new + w*h*k;
			cl_mem squared_new = (cl_mem) squared_float;
            axpy_ongpu(w*h, layer.alpha, squared_new/*squared + w*h*k*/, 1, norms, 1);
        }

        for(k = 1; k < layer.c; ++k){

			float * norms_float = (float *) norms;	
			float * new_norms_float = norms_float + w*h*(k-1);
			float * new_norms_float2 = norms_float + w*h*k;
			cl_mem new_norms = (cl_mem) new_norms_float;
			cl_mem new_norms2 = (cl_mem) new_norms_float2;

			
		
            copy_ongpu(w*h, new_norms/*norms + w*h*(k-1)*/, 1, new_norms2/*norms + w*h*k*/, 1);
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            if(prev >= 0){
				float * new = (float *) squared;
    			float *squared_float = new + w*h*prev;
				cl_mem squared_new = (cl_mem) squared_float;

				float * norms_trans = (float *) norms;
    			float *norms_float = norms_trans + w*h*k;
				cl_mem norms_new = (cl_mem) norms_float;
				axpy_ongpu(w*h, -layer.alpha, squared_new/*squared + w*h*prev*/, 1, norms_new/*norms + w*h*k*/, 1);
			}
            if(next < layer.c){
				float * new = (float *) squared;
    			float *squared_float = new + w*h*next;
				cl_mem squared_new = (cl_mem) squared_float;

				float * norms_trans = (float *) norms;
    			float *norms_float = norms_trans + w*h*k;
				cl_mem norms_new = (cl_mem) norms_float;

				axpy_ongpu(w*h,  layer.alpha, squared_new/*squared + w*h*next*/, 1, norms_new/*norms + w*h*k*/, 1);
			}
        }
    }
    pow_ongpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, layer.output_gpu, 1);
	cl_mem new_net = (cl_mem) net.input_gpu;
    mul_ongpu(w*h*c*layer.batch, new_net/*net.input_gpu*/, 1, layer.output_gpu, 1);
}

void backward_normalization_layer_gpu(const layer layer, network net)
{
    // TODO This is approximate ;-)

    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
	cl_mem new_net = (cl_mem) net.delta_gpu;
    pow_ongpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, new_net/*net.delta_gpu*/, 1);
    mul_ongpu(w*h*c*layer.batch, layer.delta_gpu, 1, new_net/*net.delta_gpu*/, 1);
}
#endif
