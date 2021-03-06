#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

//#include "cuda.h"
#include "opencl.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef GPU
void forward_deconvolutional_layer_gpu(layer l, network net);
void backward_deconvolutional_layer_gpu(layer l, network net);
void update_deconvolutional_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay);
void push_deconvolutional_layer(layer l);
void pull_deconvolutional_layer(layer l);
#endif
static size_t get_workspace_size(layer l);
layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam);
void resize_deconvolutional_layer(layer *l, int h, int w);
void forward_deconvolutional_layer(const layer l, network net);
void update_deconvolutional_layer(layer l, int batch, float learning_rate, float momentum, float decay);
void backward_deconvolutional_layer(layer l, network net);

#endif

