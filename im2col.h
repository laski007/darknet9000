#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU
#include "opencl.h"
cl_kernel get_im2col_kernel();
void im2col_ongpu(cl_mem im,int channels, int height, int width,int ksize, int stride, int pad, cl_mem data_col);


#endif
#endif
