#ifndef RESNET_CUH
#define RESNET_CUH

#include "../neural_network.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <torch/torch.h>
#include <curand.h>

// Model configuration
typedef struct {
    int input_channels;
    int input_height;
    int input_width;
    int num_filters;
    int num_residual_blocks;
    int num_actions;
    float learning_rate;
    float weight_decay;
    int batch_size;
    int epochs;
} ModelConfig;

// ResNet model
typedef struct {
    // Input block
    cudnnFilterDescriptor_t input_conv_filter;
    float *input_conv_weights, *input_bn_scale, *input_bn_bias;

    // Residual blocks
    cudnnFilterDescriptor_t *res_conv_filters;
    float **res_conv_weights, **res_bn_scales, **res_bn_biases;
    float **res_bn_means, **res_bn_vars;

    // Output block
    cudnnFilterDescriptor_t value_conv_filter, policy_conv_filter;
    float *value_conv_weights, *value_bn_scale, *value_bn_bias;
    float *value_fc1_weights, *value_fc1_bias;
    float *value_fc2_weights, *value_fc2_bias;
    float *policy_conv_weights, *policy_bn_scale, *policy_bn_bias;
    float *policy_fc_weights, *policy_fc_bias;

    // CUDNN handle
    cudnnHandle_t cudnn_handle;

    // Model configuration
    ModelConfig config;

    // Workspace for cuDNN
    void *workspace;
    size_t workspace_size;

    // Gradient fields
    float *d_input_conv_weights;
    float **d_res_conv_weights;
    float *d_value_conv_weights, *d_policy_conv_weights;

    // Additional fields
    float *value_bn_mean, *value_bn_var;
    float *policy_bn_mean, *policy_bn_var;
} ResNetModel;

typedef struct {
    INeuralNet base;
    ResNetModel model;
} ResNetWrapper;

// Function prototypes
INeuralNet* create_resnet_model(const IGame* game);
static void resnet_init(INeuralNet* self, const IGame* game);
static void resnet_train(INeuralNet* self, TrainingExample* examples, int num_examples);
static void resnet_predict(INeuralNet* self, const float* board, float* pi, float* v);
static void resnet_save_checkpoint(INeuralNet* self, const char* folder, const char* filename);
static void resnet_load_checkpoint(INeuralNet* self, const char* folder, const char* filename);
static void resnet_destroy(INeuralNet* self);

// Helper function prototypes
static void init_model_config(ResNetModel* model, const IGame* game);
static void init_input_block(ResNetModel* model);
static void init_residual_blocks(ResNetModel* model);
static void init_output_block(ResNetModel* model);
static void init_weights(ResNetModel* model);
static void prepare_batch(TrainingExample* examples, int num_examples, int batch_size,
                          float** batch_boards, float** batch_pis, float** batch_vs);
static void forward_resnet(ResNetModel* model, float* batch_boards, float** out_pi, float** out_v);
static std::pair<float, float> compute_losses(float* target_pi, float* target_v, float* out_pi, float* out_v, int batch_size, int action_size);
static void backward_resnet(ResNetModel* model, float* batch_boards, float* target_pi, float* target_v, float* out_pi, float* out_v);
static void adam_update(torch::optim::Adam& optimizer);

// CUDA kernel declarations
__global__ void policy_loss_kernel(float* target_pi, float* out_pi, float* loss, int batch_size, int action_size);
__global__ void value_loss_kernel(float* target_v, float* out_v, float* loss, int batch_size);

#endif // RESNET_CUH