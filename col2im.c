#include <stdio.h>
#include <math.h>
void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}
//This one might be too, can't remember.
void col2im_cpu(float* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, float* data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
}

#ifdef GPU
#include "opencl.h"
#define BLOCK 64
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

cl_kernel get_col2im_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/col2im_kernels.cl", "col2im_gpu_kernel", "-D BLOCK=" STR(BLOCK));
        init = 1;
    }
    return kernel;
}
 

void col2im_ongpu(cl_mem data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, cl_mem data_im){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
	//cl_mem data_col = (cl_mem) cuda_data_col;
	//cl_mem data_im = (cl_mem) cuda_data_im;
	cl_kernel kernel = get_col2im_kernel();
    cl_command_queue queue = cl.queue;

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
	cl_uint i = 0;

    cl.error = clSetKernelArg(kernel, i++, sizeof(num_kernels), (void*) &num_kernels);
    cl.error = clSetKernelArg(kernel, i++, sizeof(data_col), (void*) &data_col);
    cl.error = clSetKernelArg(kernel, i++, sizeof(height), (void*) &height);
	cl.error = clSetKernelArg(kernel, i++, sizeof(width), (void*) &width);
    cl.error = clSetKernelArg(kernel, i++, sizeof(ksize), (void*) &ksize);
    cl.error = clSetKernelArg(kernel, i++, sizeof(pad), (void*) &pad);
	cl.error = clSetKernelArg(kernel, i++, sizeof(stride), (void*) &stride);
	cl.error = clSetKernelArg(kernel, i++, sizeof(height_col), (void*) &height_col);
	cl.error = clSetKernelArg(kernel, i++, sizeof(width_col), (void*) &width_col);
	cl.error = clSetKernelArg(kernel, i++, sizeof(data_im), (void*) &data_im);
    cl_check_error(cl);

	size_t global_size[] = {num_kernels};
	size_t localws[]={BLOCK};
	cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0,
            global_size, localws, 0, 0, 0);
	cl_check_error(cl);

/*    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, data_col, height, width, ksize, pad,
                stride, height_col,
                width_col, data_im);
*/
}

#endif

