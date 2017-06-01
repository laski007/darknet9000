#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"
#include "opencl.h"

load_args get_base_args(network net)
{
    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.size = net.w;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.center = net.center;
    args.saturation = net.saturation;
    args.hue = net.hue;
    return args;
}

network load_network(char *cfg, char *weights, int clear)
{
    network net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(&net, weights);
    }
    if(clear) *net.seen = 0;
    return net;
}

int get_current_batch(network net)
{
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

void reset_momentum(network net)
{
    if (net.momentum == 0) return;
    net.learning_rate = 0;
    net.momentum = 0;
    net.decay = 0;
    #ifdef GPU
        //if(net.gpu_index >= 0) update_network_gpu(net);
    #endif
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                //if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
            return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(int));
    net.cost = calloc(1, sizeof(float));
    return net;
}

void forward_network(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, net);
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(net);
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
        }
    }
}

void calc_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network net)
{
    return max_index(net.output, net.outputs);
}

void backward_network(network net)
{
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}

float train_network_datum(network net)
{
#ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net);
#endif
    *net.seen += net.batch;
    net.train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net.cost;
    if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net.input, net.truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_network(network net, data d)
{
    assert(d.X.rows % net.batch == 0);
    int batch = net.batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net.input, net.truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    //cuda_set_device(net->gpu_index);
    //cuda_free(net->workspace);
	//cl_init(net->gpu_index);
	clReleaseMemObject(net->workspace);
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(*net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        //cuda_free(net->input_gpu);
        //cuda_free(net->truth_gpu);
		clReleaseMemObject(net->input_gpu);
		clReleaseMemObject(net->truth_gpu);
        net->input_gpu = cl_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cl_make_array(net->truth, net->truths*net->batch);
        net->workspace = cl_make_array(0, (workspace_size-1)/sizeof(float)+1);
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

detection_layer get_network_detection_layer(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].type == DETECTION){
            return net.layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    detection_layer l = {0};
    return l;
}

image get_network_image_layer(network net, int i)
{
    layer l = net.layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    }
}

void top_predictions(network net, int k, int *index)
{
    top_k(net.output, net.outputs, k, index);
}


float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif
    net.input = input;
    net.truth = 0;
    net.train = 0;
    net.delta = 0;
    forward_network(net);
    return net.output;
}

matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = net.outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net.batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = net.outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;
}

void print_network(network net)
{
    int i,j;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den);
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network net)
{
    int i;
    for(i = net.n - 1; i >= 0; --i){
        if(net.layers[i].type != COST) break;
    }
    return net.layers[i];
}

float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        free_layer(net.layers[i]);
    }
    free(net.layers);
    if(net.input) free(net.input);
    if(net.truth) free(net.truth);
#ifdef GPU
    if(net.input_gpu) clReleaseMemObject(net.input_gpu);//cuda_free(net.input_gpu);
    if(net.truth_gpu) clReleaseMemObject(net.truth_gpu);//cuda_free(net.truth_gpu);
#endif
}

//move the files from network_kernels.cu to here, because cuda is no use any more.
void forward_network_gpu(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            size_t delta_g;
            clGetMemObjectInfo(l.delta_gpu, CL_MEM_SIZE, sizeof(size_t), &delta_g, NULL);
            printf("%d",sizeof(float));
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(net);
    calc_network_cost(net);
}

void backward_network_gpu(network net)
{
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void update_network_gpu(network net)
{
    //cuda_set_device(net.gpu_index);
	//cl_init(net.gpu_index);
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        l.t = get_current_batch(net);
        if(l.update_gpu){
            l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
        }
    }
}

void harmless_update_network_gpu(network net)
{
    //cuda_set_device(net.gpu_index);
	//cl_init(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_ongpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_ongpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_ongpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

float train_network_datum_gpu(network net)
{
    *net.seen += net.batch;

    int x_size = net.inputs*net.batch;
    int y_size = net.truths*net.batch;
    cl_write_array(net.input_gpu, net.input, x_size);
    cl_write_array(net.truth_gpu, net.truth, y_size);

    net.train = 1;
    forward_network_gpu(net);
    backward_network_gpu(net);

    float error = *net.cost;
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);

    return error;
}
/*
typedef struct {
    network net;
    data d;
    float *err;
} train_args;
*/
void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    //cuda_set_device(args.net.gpu_index);
	//cl_init(args.net.gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cl_read_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cl_read_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cl_read_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cl_read_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cl_read_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cl_write_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cl_write_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cl_write_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cl_write_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cl_write_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void update_layer(layer l, network net)
{
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    l.t = get_current_batch(net);
    if(l.update_gpu){
        l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
    }
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weights, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.n*l.size*l.size*l.c, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}

void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cl_read_array(l.biases_gpu, l.biases, l.n);
        cl_read_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cl_read_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cl_read_array(l.biases_gpu, l.biases, l.outputs);
        cl_read_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cl_write_array(l.biases_gpu, l.biases, l.n);
        cl_write_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cl_write_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cl_write_array(l.biases_gpu, l.biases, l.outputs);
        cl_write_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cl_write_array(l.biases_gpu, base.biases, l.n);
        cl_write_array(l.weights_gpu, base.weights, l.n*l.size*l.size*l.c);
        if(base.scales) cl_write_array(l.scales_gpu, base.scales, l.n);
    } else if(l.type == CONNECTED){
        cl_write_array(l.biases_gpu, base.biases, l.outputs);
        cl_write_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}

void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weight_updates, 1, base.weight_updates, 1);
        if (l.scale_updates) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
    }
}

void distribute_updates(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cl_write_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cl_write_array(l.weight_updates_gpu, base.weight_updates, l.n*l.size*l.size*l.c);
        if(base.scale_updates) cl_write_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cl_write_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cl_write_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}

void sync_layer(network *nets, int n, int j)
{
    //printf("Syncing layer %d\n", j);
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    //cuda_set_device(net.gpu_index);
	//cl_init(net.gpu_index);
    pull_weights(base);
    for (i = 1; i < n; ++i) {
        //cuda_set_device(nets[i].gpu_index);
		//cl_init(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        //cuda_set_device(nets[i].gpu_index);
		//cl_init(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
    //printf("Done syncing layer %d\n", j);
}
/*
typedef struct{
    network *nets;
    int n;
    int j;
} sync_args;
*/
void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network *nets, int n, int interval)
{
    int j;
    int layers = nets[0].n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
    for (j = 0; j < n; ++j){
        *nets[j].seen = *nets[0].seen;
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network *nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network net)
{
    layer l = get_network_output_layer(net);
    cl_read_array(l.output_gpu, l.output, l.outputs*l.batch);
}

float *network_predict_gpu(network net, float *input)
{
    //cuda_set_device(net.gpu_index);
	//cl_init(net.gpu_index);
    cl_write_array(net.input_gpu, input, net.inputs*net.batch);
    net.truth = 0;
    net.train = 0;
    forward_network_gpu(net);
    return net.output;
}
//end of moving

// Some day...


layer network_output_layer(network net)
{
    int i;
    for(i = net.n - 1; i >= 0; --i){
        if(net.layers[i].type != COST) break;
    }
    return net.layers[i];
}

int network_inputs(network net)
{
    return net.layers[0].inputs;
}

int network_outputs(network net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network net)
{
    return network_output_layer(net).output;
}

