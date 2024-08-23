#ifndef GAT_CUH
#define GAT_CUH

#include "../neural_network.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <torch/torch.h>

// Model configuration
typedef struct {
    int input_features;
    int hidden_features;
    int output_features;
    int num_heads;
    int num_layers;
    int num_actions;
    int max_nodes;
    int max_edges;
    float learning_rate;
    float weight_decay;
    float dropout;
    float alpha;  // LeakyReLU angle
    int batch_size;
    int epochs;
} ModelConfig;

// GAT model
typedef struct {
    // Input block
    cudnnTensorDescriptor_t input_descriptor;
    float *input_weights, *input_bias;

    // GAT layers
    cudnnTensorDescriptor_t *layer_descriptors;
    float **layer_weights, **layer_biases;
    float **attention_weights;

    // Output block
    cudnnTensorDescriptor_t value_descriptor, policy_descriptor;
    float *value_weights, *value_bias;
    float *policy_weights, *policy_bias;

    // Running mean and variance for batch normalization (if used)
    float *input_bn_mean, *input_bn_var;
    float **layer_bn_means, **layer_bn_vars;
    float *value_bn_mean, *value_bn_var, *policy_bn_mean, *policy_bn_var;

    // Saved mean and variance for batch normalization (if used)
    float *input_bn_save_mean, *input_bn_save_var;
    float **layer_bn_save_means, **layer_bn_save_vars;
    float *value_bn_save_mean, *value_bn_save_var, *policy_bn_save_mean, *policy_bn_save_var;

    // CUDNN & CUBLAS handle
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;


    // Model configuration
    ModelConfig config;

    // PyTorch optimizer
    torch::optim::Adam* optimizer;

    // Gradient fields
    float *d_input_weights;
    float **d_layer_weights, **d_attention_weights;
    float *d_value_weights, *d_policy_weights;

    // Workspace for cuDNN
    void *workspace;
    size_t workspace_size;

    // Learning rate and weight decay
    float learning_rate;
    float weight_decay;
} GATModel;

typedef struct {
    INeuralNet base;
    GATModel model;
} GATWrapper;

// Function prototypes
INeuralNet* create_gat_model(const IGame* game);
static void gat_init(INeuralNet* self, const IGame* game);
static void gat_train(INeuralNet* self, TrainingExample* examples, int num_examples);
static void gat_predict(INeuralNet* self, const float* board, float* pi, float* v);
static void gat_save_checkpoint(INeuralNet* self, const char* folder, const char* filename);
static void gat_load_checkpoint(INeuralNet* self, const char* folder, const char* filename);
static void gat_destroy(INeuralNet* self);

// Distributed stuff
static void gat_train_distributed(INeuralNet* self, float* d_boards, float* d_pis, float* d_vs, int num_examples, int world_rank, int world_size, ncclComm_t nccl_comm, cudaStream_t cuda_stream);
static void gat_broadcast_weights(INeuralNet* self, int world_rank, int world_size, ncclComm_t nccl_comm, cudaStream_t cuda_stream);

// Helper function prototypes
static void init_model_config(GATModel* model, const IGame* game);
static void init_input_block(GATModel* model);
static void init_gat_layers(GATModel* model);
static void init_output_block(GATModel* model);
static void init_weights(GATModel* model);
static void prepare_batch(TrainingExample* examples, int num_examples, int batch_size, float** batch_boards, float** batch_pis, float** batch_vs);
static void forward_gat(GATModel* model, float* batch_boards, float** out_pi, float** out_v);
static std::pair<float, float> compute_losses(float* target_pi, float* target_v, float* out_pi, float* out_v, int batch_size, int action_size);
static void backward_gat(GATModel* model, float* batch_boards, float* target_pi, float* target_v, float* out_pi, float* out_v);
static void adam_update(torch::optim::Adam& optimizer);
static void compute_initial_grad_output(GATModel* model, float* d_policy, float* d_value, float* grad_output, int batch_size);


#endif // GAT_CUH