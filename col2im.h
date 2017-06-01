#ifndef COL2IM_H
#define COL2IM_H

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU
cl_kernel get_col2im_kernel();
void col2im_ongpu(cl_mem data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, cl_mem data_im);
#endif
#endif
