#include "crop_layer.h"
//#include "cuda.h"
#include "opencl.h"
#include <stdio.h>

image get_crop_image(crop_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return float_to_image(w,h,c,l.output);
}

void backward_crop_layer(const crop_layer l, network net){}
void backward_crop_layer_gpu(const crop_layer l, network net){}

crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
{
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer l = {0};
    l.type = CROP;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.scale = (float)crop_height / h;
    l.flip = flip;
    l.angle = angle;
    l.saturation = saturation;
    l.exposure = exposure;
    l.out_w = crop_width;
    l.out_h = crop_height;
    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = calloc(l.outputs*batch, sizeof(float));
    l.forward = forward_crop_layer;
    l.backward = backward_crop_layer;

    #ifdef GPU
    l.forward_gpu = forward_crop_layer_gpu;
    l.backward_gpu = backward_crop_layer_gpu;
    l.output_gpu = cl_make_array(l.output, l.outputs*batch);
    l.rand_gpu   = cl_make_array(0, l.batch*8);
    #endif
    return l;
}

void resize_crop_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->out_w =  l->scale*w;
    l->out_h =  l->scale*h;

    l->inputs = l->w * l->h * l->c;
    l->outputs = l->out_h * l->out_w * l->out_c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    #ifdef GPU
    //cuda_free(l->output_gpu);
	clReleaseMemObject(l->output_gpu);
    l->output_gpu = cl_make_array(l->output, l->outputs*l->batch);
    #endif
}


void forward_crop_layer(const crop_layer l, network net)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
    int flip = (l.flip && rand()%2);
    int dh = rand()%(l.h - l.out_h + 1);
    int dw = rand()%(l.w - l.out_w + 1);
    float scale = 2;
    float trans = -1;
    if(l.noadjust){
        scale = 1;
        trans = 0;
    }
    if(!net.train){
        flip = 0;
        dh = (l.h - l.out_h)/2;
        dw = (l.w - l.out_w)/2;
    }
    for(b = 0; b < l.batch; ++b){
        for(c = 0; c < l.c; ++c){
            for(i = 0; i < l.out_h; ++i){
                for(j = 0; j < l.out_w; ++j){
                    if(flip){
                        col = l.w - dw - j - 1;    
                    }else{
                        col = j + dw;
                    }
                    row = i + dh;
                    index = col+l.w*(row+l.h*(c + l.c*b)); 
                    l.output[count++] = net.input[index]*scale + trans;
                }
            }
        }
    }
}

#ifdef GPU
#define BLOCK 64
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
cl_kernel get_levels_image_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/crop_layer_kernels.cl", "levels_image_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

cl_kernel get_forward_crop_layer_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/crop_layer_kernels.cl", "forward_crop_layer_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}

void forward_crop_layer_gpu(crop_layer layer, network net)
{

	cl_random(layer.rand_gpu, layer.batch*8);
	float radians = layer.angle*3.14159265/180.;

	float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;

    cl_kernel kernel1 = get_levels_image_kernel();
    cl_command_queue queue1 = cl.queue;
	
    cl_uint i = 0;

    cl.error = clSetKernelArg(kernel1, i++, sizeof(net.input_gpu), (void*) &(net.input_gpu));
    cl.error = clSetKernelArg(kernel1, i++, sizeof(layer.rand_gpu), (void*) &(layer.rand_gpu));
	cl.error = clSetKernelArg(kernel1, i++, sizeof(layer.batch), (void*) &(layer.batch));
    cl.error = clSetKernelArg(kernel1, i++, sizeof(layer.w), (void*) &(layer.w));
	cl.error = clSetKernelArg(kernel1, i++, sizeof(layer.h), (void*) &(layer.h));
	cl.error = clSetKernelArg(kernel1, i++, sizeof(net.train), (void*) &(net.train));
	cl.error = clSetKernelArg(kernel1, i++, sizeof(layer.saturation), (void*) &(layer.saturation));
	cl.error = clSetKernelArg(kernel1, i++, sizeof(layer.exposure), (void*) &(layer.exposure));
	cl.error = clSetKernelArg(kernel1, i++, sizeof(translate), (void*) &translate);
	cl.error = clSetKernelArg(kernel1, i++, sizeof(scale), (void*) &scale);
	cl.error = clSetKernelArg(kernel1, i++, sizeof(layer.shift), (void*) &(layer.shift));

	cl_check_error(cl);

    const size_t gsize1[] = {size*BLOCK};
    const size_t localws1[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue1, kernel1, 1, 0, gsize1, localws1, 0, 0, 0);
    cl_check_error(cl);
//load the 2nd kernel
	cl_kernel kernel2 = get_forward_crop_layer_kernel();
    cl_command_queue queue2 = cl.queue;
	
    cl_uint j = 0;

    cl.error = clSetKernelArg(kernel2, j++, sizeof(net.input_gpu), (void*) &(net.input_gpu));
    cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.rand_gpu), (void*) &(layer.rand_gpu));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(size), (void*) &size);
    cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.c), (void*) &(layer.c));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.h), (void*) &(layer.h));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.w), (void*) &(layer.w));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.out_h), (void*) &(layer.out_h));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.out_w), (void*) &(layer.out_w));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(net.train), (void*) &(net.train));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.flip), (void*) &(layer.flip));
	cl.error = clSetKernelArg(kernel2, j++, sizeof(radians), (void*) &radians);
	cl.error = clSetKernelArg(kernel2, j++, sizeof(layer.output_gpu), (void*) &(layer.output_gpu));

	cl_check_error(cl);

    const size_t gsize2[] = {size*BLOCK};
    const size_t localws2[] = {BLOCK};//localws[1] = [64];
    cl.error = clEnqueueNDRangeKernel(queue2, kernel2, 1, 0, gsize2, localws2, 0, 0, 0);
    cl_check_error(cl);

}

#endif

