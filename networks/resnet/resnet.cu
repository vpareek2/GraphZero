#include "resnet.cuh"

#include <curand.h>
#include <cublas_v2.h>
#include <torch/torch.h>

static void resnet_init(INeuralNet* self, const IGame* game) {
    ResNetWrapper* wrapper = (ResNetWrapper*)self;
    ResNetModel* model = &wrapper->model;
    
    // Initialize model configuration based on game
    init_model_config(model, game);

    // Initialize cuDNN
    cudnnCreate(&model->cudnn_handle);

    // Initialize input block
    init_input_block(model);

    // Initialize residual blocks
    init_residual_blocks(model);

    // Initialize output block
    init_output_block(model);

    // Initialize weights with small random values
    init_weights(model);
}

static void resnet_train(INeuralNet* self, TrainingExample* examples, int num_examples) {
    ResNetWrapper* wrapper = (ResNetWrapper*)self;
    ResNetModel* model = &wrapper->model;

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Create Adam optimizer
    AdamOptimizer optimizer;
    init_adam_optimizer(&optimizer, model->learning_rate, model->weight_decay);

    // Training loop
    for (int epoch = 0; epoch < model->config.epochs; epoch++) {
        printf("EPOCH ::: %d\n", epoch + 1);

        float pi_loss_sum = 0.0f;
        float v_loss_sum = 0.0f;
        int batch_count = num_examples / model->config.batch_size;

        for (int batch = 0; batch < batch_count; batch++) {
            // Prepare batch data
            float* batch_boards;
            float* batch_pis;
            float* batch_vs;
            prepare_batch(examples, num_examples, model->config.batch_size, 
                          &batch_boards, &batch_pis, &batch_vs);

            // Forward pass
            float* out_pi;
            float* out_v;
            forward_resnet(model, batch_boards, &out_pi, &out_v);

            // Compute losses
            float l_pi = compute_policy_loss(batch_pis, out_pi, model->config.batch_size, model->config.action_size);
            float l_v = compute_value_loss(batch_vs, out_v, model->config.batch_size);
            float total_loss = l_pi + l_v;

            // Backward pass
            backward_resnet(model, batch_boards, batch_pis, batch_vs, out_pi, out_v);

            // Update weights
            adam_update(&optimizer, model);

            // Record loss
            pi_loss_sum += l_pi;
            v_loss_sum += l_v;

            // Clean up
            cudaFree(batch_boards);
            cudaFree(batch_pis);
            cudaFree(batch_vs);
            cudaFree(out_pi);
            cudaFree(out_v);
        }

        // Print epoch results
        printf("Average Policy Loss: %f, Average Value Loss: %f\n", 
               pi_loss_sum / batch_count, v_loss_sum / batch_count);
    }

    // Clean up
    cublasDestroy(cublas_handle);
}

static void resnet_predict(INeuralNet* self, const float* board, float* pi, float* v) {
    ResNetWrapper* wrapper = (ResNetWrapper*)self;
    ResNetModel* model = &wrapper->model;

    // Allocate device memory for input and output
    float *d_board, *d_pi, *d_v;
    cudaMalloc(&d_board, sizeof(float) * model->config.input_channels * model->config.input_height * model->config.input_width);
    cudaMalloc(&d_pi, sizeof(float) * model->config.num_actions);
    cudaMalloc(&d_v, sizeof(float));

    // Copy input to device
    cudaMemcpy(d_board, board, sizeof(float) * model->config.input_channels * model->config.input_height * model->config.input_width, cudaMemcpyHostToDevice);

    // Forward pass
    forward_resnet(model, d_board, &d_pi, &d_v);

    // Copy output back to host
    cudaMemcpy(pi, d_pi, sizeof(float) * model->config.num_actions, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_board);
    cudaFree(d_pi);
    cudaFree(d_v);
}

static void resnet_save_checkpoint(INeuralNet* self, const char* folder, const char* filename) {
    ResNetWrapper* wrapper = (ResNetWrapper*)self;
    ResNetModel* model = &wrapper->model;

    char filepath[MAX_FILENAME_LENGTH];
    snprintf(filepath, MAX_FILENAME_LENGTH, "%s/%s", folder, filename);

    FILE* file = fopen(filepath, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file for writing: %s\n", filepath);
        return;
    }

    // Save model configuration
    fwrite(&model->config, sizeof(ModelConfig), 1, file);

    // Save weights
    // Input block
    fwrite(model->input_conv_weights, sizeof(float), model->config.num_filters * model->config.input_channels * 3 * 3, file);
    fwrite(model->input_bn_scale, sizeof(float), model->config.num_filters, file);
    fwrite(model->input_bn_bias, sizeof(float), model->config.num_filters, file);

    // Residual blocks
    for (int i = 0; i < model->config.num_residual_blocks * 2; i++) {
        fwrite(model->res_conv_weights[i], sizeof(float), model->config.num_filters * model->config.num_filters * 3 * 3, file);
        fwrite(model->res_bn_scales[i], sizeof(float), model->config.num_filters, file);
        fwrite(model->res_bn_biases[i], sizeof(float), model->config.num_filters, file);
    }

    // Output block
    fwrite(model->value_conv_weights, sizeof(float), model->config.num_filters, file);
    fwrite(model->value_bn_scale, sizeof(float), 1, file);
    fwrite(model->value_bn_bias, sizeof(float), 1, file);
    fwrite(model->value_fc1_weights, sizeof(float), model->config.input_height * model->config.input_width * 256, file);
    fwrite(model->value_fc1_bias, sizeof(float), 256, file);
    fwrite(model->value_fc2_weights, sizeof(float), 256, file);
    fwrite(model->value_fc2_bias, sizeof(float), 1, file);

    fwrite(model->policy_conv_weights, sizeof(float), 2 * model->config.num_filters, file);
    fwrite(model->policy_bn_scale, sizeof(float), 2, file);
    fwrite(model->policy_bn_bias, sizeof(float), 2, file);
    fwrite(model->policy_fc_weights, sizeof(float), 2 * model->config.input_height * model->config.input_width * model->config.num_actions, file);
    fwrite(model->policy_fc_bias, sizeof(float), model->config.num_actions, file);

    fclose(file);
}

static void resnet_load_checkpoint(INeuralNet* self, const char* folder, const char* filename) {
    ResNetWrapper* wrapper = (ResNetWrapper*)self;
    ResNetModel* model = &wrapper->model;

    char filepath[MAX_FILENAME_LENGTH];
    snprintf(filepath, MAX_FILENAME_LENGTH, "%s/%s", folder, filename);

    FILE* file = fopen(filepath, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file for reading: %s\n", filepath);
        return;
    }

    // Load model configuration
    fread(&model->config, sizeof(ModelConfig), 1, file);

    // Load weights
    // Input block
    cudaMemcpy(model->input_conv_weights, model->input_conv_weights, sizeof(float) * model->config.num_filters * model->config.input_channels * 3 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(model->input_bn_scale, model->input_bn_scale, sizeof(float) * model->config.num_filters, cudaMemcpyHostToDevice);
    cudaMemcpy(model->input_bn_bias, model->input_bn_bias, sizeof(float) * model->config.num_filters, cudaMemcpyHostToDevice);

    // Residual blocks
    for (int i = 0; i < model->config.num_residual_blocks * 2; i++) {
        cudaMemcpy(model->res_conv_weights[i], model->res_conv_weights[i], sizeof(float) * model->config.num_filters * model->config.num_filters * 3 * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(model->res_bn_scales[i], model->res_bn_scales[i], sizeof(float) * model->config.num_filters, cudaMemcpyHostToDevice);
        cudaMemcpy(model->res_bn_biases[i], model->res_bn_biases[i], sizeof(float) * model->config.num_filters, cudaMemcpyHostToDevice);
    }

    // Output block
    cudaMemcpy(model->value_conv_weights, model->value_conv_weights, sizeof(float) * model->config.num_filters, cudaMemcpyHostToDevice);
    cudaMemcpy(model->value_bn_scale, model->value_bn_scale, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(model->value_bn_bias, model->value_bn_bias, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(model->value_fc1_weights, model->value_fc1_weights, sizeof(float) * model->config.input_height * model->config.input_width * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(model->value_fc1_bias, model->value_fc1_bias, sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(model->value_fc2_weights, model->value_fc2_weights, sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(model->value_fc2_bias, model->value_fc2_bias, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(model->policy_conv_weights, model->policy_conv_weights, sizeof(float) * 2 * model->config.num_filters, cudaMemcpyHostToDevice);
    cudaMemcpy(model->policy_bn_scale, model->policy_bn_scale, sizeof(float) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(model->policy_bn_bias, model->policy_bn_bias, sizeof(float) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(model->policy_fc_weights, model->policy_fc_weights, sizeof(float) * 2 * model->config.input_height * model->config.input_width * model->config.num_actions, cudaMemcpyHostToDevice);
    cudaMemcpy(model->policy_fc_bias, model->policy_fc_bias, sizeof(float) * model->config.num_actions, cudaMemcpyHostToDevice);

    fclose(file);
}

static void resnet_destroy(INeuralNet* self) {
    ResNetWrapper* wrapper = (ResNetWrapper*)self;
    ResNetModel* model = &wrapper->model;

    // Free device memory
    cudaFree(model->input_conv_weights);
    cudaFree(model->input_bn_scale);
    cudaFree(model->input_bn_bias);

    for (int i = 0; i < model->config.num_residual_blocks * 2; i++) {
        cudaFree(model->res_conv_weights[i]);
        cudaFree(model->res_bn_scales[i]);
        cudaFree(model->res_bn_biases[i]);
    }

    cudaFree(model->value_conv_weights);
    cudaFree(model->value_bn_scale);
    cudaFree(model->value_bn_bias);
    cudaFree(model->value_fc1_weights);
    cudaFree(model->value_fc1_bias);
    cudaFree(model->value_fc2_weights);
    cudaFree(model->value_fc2_bias);

    cudaFree(model->policy_conv_weights);
    cudaFree(model->policy_bn_scale);
    cudaFree(model->policy_bn_bias);
    cudaFree(model->policy_fc_weights);
    cudaFree(model->policy_fc_bias);

    // Free host memory
    free(model->res_conv_filters);
    free(model->res_bn_means);
    free(model->res_bn_vars);
    free(model->res_conv_weights);
    free(model->res_bn_scales);
    free(model->res_bn_biases);

    // Destroy cuDNN handles
    cudnnDestroy(model->cudnn_handle);

    // Free the wrapper
    free(wrapper);
}

INeuralNet* create_resnet_model(const IGame* game) {
    ResNetWrapper* wrapper = (ResNetWrapper*)malloc(sizeof(ResNetWrapper));
    wrapper->base.impl = wrapper;
    wrapper->base.init = resnet_init;
    wrapper->base.train = resnet_train;
    wrapper->base.predict = resnet_predict;
    wrapper->base.save_checkpoint = resnet_save_checkpoint;
    wrapper->base.load_checkpoint = resnet_load_checkpoint;
    wrapper->base.destroy = resnet_destroy;

    resnet_init(&wrapper->base, game);

    return &wrapper->base;
}
/*************************************************************************************************************************************************************
 * INIT HELPER FUNCTIONS
**************************************************************************************************************************************************************/

static void init_model_config(ResNetModel* model, const IGame* game) {
    // Set up model configuration based on game parameters
    int rows, cols;
    game->get_board_size(game, &rows, &cols);
    model->config.input_channels = 3;  // Assuming 3 channels for player 1, player 2, and turn
    model->config.input_height = rows;
    model->config.input_width = cols;
    model->config.num_actions = game->get_action_size(game);
    model->config.num_residual_blocks = 19;  // AlphaZero used 19 residual blocks
    model->config.num_filters = 256;
    model->config.learning_rate = 0.001;
    model->config.weight_decay = 0.0001;
}

static void init_input_block(ResNetModel* model) {
    // Create and initialize convolution filter descriptor
    cudnnCreateFilterDescriptor(&model->input_conv_filter);
    cudnnSetFilter4dDescriptor(model->input_conv_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               model->config.num_filters, model->config.input_channels,
                               3, 3);  // 3x3 convolution

    // Allocate memory for convolution weights
    cudaMalloc(&model->input_conv_weights, sizeof(float) * model->config.num_filters * model->config.input_channels * 3 * 3);

    // Create and initialize batch normalization descriptors
    cudnnCreateTensorDescriptor(&model->input_bn_mean);
    cudnnCreateTensorDescriptor(&model->input_bn_var);
    cudnnSetTensor4dDescriptor(model->input_bn_mean, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, model->config.num_filters, 1, 1);
    cudnnSetTensor4dDescriptor(model->input_bn_var, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, model->config.num_filters, 1, 1);

    // Allocate memory for batch normalization parameters
    cudaMalloc(&model->input_bn_scale, sizeof(float) * model->config.num_filters);
    cudaMalloc(&model->input_bn_bias, sizeof(float) * model->config.num_filters);
}

static void init_residual_blocks(ResNetModel* model) {
    // Allocate arrays for residual block parameters
    model->res_conv_filters = malloc(sizeof(cudnnFilterDescriptor_t) * model->config.num_residual_blocks * 2);
    model->res_bn_means = malloc(sizeof(cudnnTensorDescriptor_t) * model->config.num_residual_blocks * 2);
    model->res_bn_vars = malloc(sizeof(cudnnTensorDescriptor_t) * model->config.num_residual_blocks * 2);
    model->res_conv_weights = malloc(sizeof(float*) * model->config.num_residual_blocks * 2);
    model->res_bn_scales = malloc(sizeof(float*) * model->config.num_residual_blocks * 2);
    model->res_bn_biases = malloc(sizeof(float*) * model->config.num_residual_blocks * 2);

    for (int i = 0; i < model->config.num_residual_blocks; i++) {
        for (int j = 0; j < 2; j++) {
            int idx = i * 2 + j;
            // Create and initialize convolution filter descriptor
            cudnnCreateFilterDescriptor(&model->res_conv_filters[idx]);
            cudnnSetFilter4dDescriptor(model->res_conv_filters[idx], CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                       model->config.num_filters, model->config.num_filters,
                                       3, 3);  // 3x3 convolution

            // Allocate memory for convolution weights
            cudaMalloc(&model->res_conv_weights[idx], sizeof(float) * model->config.num_filters * model->config.num_filters * 3 * 3);

            // Create and initialize batch normalization descriptors
            cudnnCreateTensorDescriptor(&model->res_bn_means[idx]);
            cudnnCreateTensorDescriptor(&model->res_bn_vars[idx]);
            cudnnSetTensor4dDescriptor(model->res_bn_means[idx], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       1, model->config.num_filters, 1, 1);
            cudnnSetTensor4dDescriptor(model->res_bn_vars[idx], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       1, model->config.num_filters, 1, 1);

            // Allocate memory for batch normalization parameters
            cudaMalloc(&model->res_bn_scales[idx], sizeof(float) * model->config.num_filters);
            cudaMalloc(&model->res_bn_biases[idx], sizeof(float) * model->config.num_filters);
        }
    }
}

static void init_output_block(ResNetModel* model) {
    // Initialize value head
    cudnnCreateFilterDescriptor(&model->value_conv_filter);
    cudnnSetFilter4dDescriptor(model->value_conv_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               1, model->config.num_filters, 1, 1);  // 1x1 convolution
    cudaMalloc(&model->value_conv_weights, sizeof(float) * model->config.num_filters);

    cudnnCreateTensorDescriptor(&model->value_bn_mean);
    cudnnCreateTensorDescriptor(&model->value_bn_var);
    cudnnSetTensor4dDescriptor(model->value_bn_mean, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 1, 1, 1);
    cudnnSetTensor4dDescriptor(model->value_bn_var, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 1, 1, 1);

    cudaMalloc(&model->value_bn_scale, sizeof(float));
    cudaMalloc(&model->value_bn_bias, sizeof(float));

    // Allocate memory for fully connected layers in value head
    int fc1_size = model->config.input_height * model->config.input_width;
    cudaMalloc(&model->value_fc1_weights, sizeof(float) * fc1_size * 256);
    cudaMalloc(&model->value_fc1_bias, sizeof(float) * 256);
    cudaMalloc(&model->value_fc2_weights, sizeof(float) * 256);
    cudaMalloc(&model->value_fc2_bias, sizeof(float));

    // Initialize policy head
    cudnnCreateFilterDescriptor(&model->policy_conv_filter);
    cudnnSetFilter4dDescriptor(model->policy_conv_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               2, model->config.num_filters, 1, 1);  // 1x1 convolution
    cudaMalloc(&model->policy_conv_weights, sizeof(float) * 2 * model->config.num_filters);

    cudnnCreateTensorDescriptor(&model->policy_bn_mean);
    cudnnCreateTensorDescriptor(&model->policy_bn_var);
    cudnnSetTensor4dDescriptor(model->policy_bn_mean, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 2, 1, 1);
    cudnnSetTensor4dDescriptor(model->policy_bn_var, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, 2, 1, 1);

    cudaMalloc(&model->policy_bn_scale, sizeof(float) * 2);
    cudaMalloc(&model->policy_bn_bias, sizeof(float) * 2);

    // Allocate memory for fully connected layer in policy head
    int policy_fc_size = 2 * model->config.input_height * model->config.input_width;
    cudaMalloc(&model->policy_fc_weights, sizeof(float) * policy_fc_size * model->config.num_actions);
    cudaMalloc(&model->policy_fc_bias, sizeof(float) * model->config.num_actions);
}

static void init_weights(ResNetModel* model) {
    // Initialize weights with small random values
    // You can use cuRAND for this purpose
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    // Initialize input block weights
    curandGenerateNormal(gen, model->input_conv_weights, model->config.num_filters * model->config.input_channels * 3 * 3, 0, 0.1);
    curandGenerateNormal(gen, model->input_bn_scale, model->config.num_filters, 1, 0.1);
    curandGenerateNormal(gen, model->input_bn_bias, model->config.num_filters, 0, 0.1);

    // Initialize residual block weights
    for (int i = 0; i < model->config.num_residual_blocks * 2; i++) {
        curandGenerateNormal(gen, model->res_conv_weights[i], model->config.num_filters * model->config.num_filters * 3 * 3, 0, 0.1);
        curandGenerateNormal(gen, model->res_bn_scales[i], model->config.num_filters, 1, 0.1);
        curandGenerateNormal(gen, model->res_bn_biases[i], model->config.num_filters, 0, 0.1);
    }

    // Initialize output block weights
    curandGenerateNormal(gen, model->value_conv_weights, model->config.num_filters, 0, 0.1);
    curandGenerateNormal(gen, model->value_bn_scale, 1, 1, 0.1);
    curandGenerateNormal(gen, model->value_bn_bias, 1, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc1_weights, model->config.input_height * model->config.input_width * 256, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc1_bias, 256, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc2_weights, 256, 0, 0.1);
    curandGenerateNormal(gen, model->value_fc2_bias, 1, 0, 0.1);

    curandGenerateNormal(gen, model->policy_conv_weights, 2 * model->config.num_filters, 0, 0.1);
    curandGenerateNormal(gen, model->policy_bn_scale, 2, 1, 0.1);
    curandGenerateNormal(gen, model->policy_bn_bias, 2, 0, 0.1);
    curandGenerateNormal(gen, model->policy_fc_weights, 2 * model->config.input_height * model->config.input_width * model->config.num_actions, 0, 0.1);
    curandGenerateNormal(gen, model->policy_fc_bias, model->config.num_actions, 0, 0.1);

    curandDestroyGenerator(gen);
}

/*************************************************************************************************************************************************************
 * TRAIN HELPER FUNCTIONS
**************************************************************************************************************************************************************/

torch::optim::Adam init_adam_optimizer(ResNetModel* model, float learning_rate, float weight_decay) {
    std::vector<torch::Tensor> params;
    
    // Input block
    params.push_back(torch::from_blob(model->input_conv_weights, {model->config.num_filters, model->config.input_channels, 3, 3}, torch::kCUDA));
    params.push_back(torch::from_blob(model->input_bn_scale, {model->config.num_filters}, torch::kCUDA));
    params.push_back(torch::from_blob(model->input_bn_bias, {model->config.num_filters}, torch::kCUDA));

    // Residual blocks
    for (int i = 0; i < model->config.num_residual_blocks * 2; i++) {
        params.push_back(torch::from_blob(model->res_conv_weights[i], {model->config.num_filters, model->config.num_filters, 3, 3}, torch::kCUDA));
        params.push_back(torch::from_blob(model->res_bn_scales[i], {model->config.num_filters}, torch::kCUDA));
        params.push_back(torch::from_blob(model->res_bn_biases[i], {model->config.num_filters}, torch::kCUDA));
    }

    // Value head
    params.push_back(torch::from_blob(model->value_conv_weights, {1, model->config.num_filters, 1, 1}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_bn_scale, {1}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_bn_bias, {1}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_fc1_weights, {256, model->config.input_height * model->config.input_width}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_fc1_bias, {256}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_fc2_weights, {1, 256}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_fc2_bias, {1}, torch::kCUDA));

    // Policy head
    params.push_back(torch::from_blob(model->policy_conv_weights, {2, model->config.num_filters, 1, 1}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_bn_scale, {2}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_bn_bias, {2}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_fc_weights, {model->config.num_actions, 2 * model->config.input_height * model->config.input_width}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_fc_bias, {model->config.num_actions}, torch::kCUDA));

    return torch::optim::Adam(params, torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
}

void prepare_batch(TrainingExample* examples, int num_examples, int batch_size,
                   float** batch_boards, float** batch_pis, float** batch_vs) {
    // Allocate memory for batch data
    cudaMalloc(batch_boards, batch_size * BOARD_SIZE * sizeof(float));
    cudaMalloc(batch_pis, batch_size * ACTION_SIZE * sizeof(float));
    cudaMalloc(batch_vs, batch_size * sizeof(float));

    // Use cuRAND for random selection
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    int* d_indices;
    cudaMalloc(&d_indices, batch_size * sizeof(int));
    curandGenerate(gen, (unsigned int*)d_indices, batch_size);

    // Custom CUDA kernel to prepare batch
    prepare_batch_kernel<<<(batch_size + 255) / 256, 256>>>(
        examples, num_examples, d_indices, *batch_boards, *batch_pis, *batch_vs, batch_size);

    cudaFree(d_indices);
    curandDestroyGenerator(gen);
}

__global__ void prepare_batch_kernel(TrainingExample* examples, int num_examples, int* indices,
                                     float* batch_boards, float* batch_pis, float* batch_vs, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int example_idx = indices[idx] % num_examples;
        memcpy(batch_boards + idx * BOARD_SIZE, examples[example_idx].board, BOARD_SIZE * sizeof(float));
        memcpy(batch_pis + idx * ACTION_SIZE, examples[example_idx].pi, ACTION_SIZE * sizeof(float));
        batch_vs[idx] = examples[example_idx].v;
    }
}

void forward_resnet(ResNetModel* model, float* batch_boards, float** out_pi, float** out_v) {
    cudnnHandle_t cudnn = model->cudnn_handle;
    float alpha = 1.0f, beta = 0.0f;
    
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnTensorDescriptor_t bn_descriptor;
    
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnCreateActivationDescriptor(&activation_descriptor);
    cudnnCreateTensorDescriptor(&bn_descriptor);
    
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, model->config.input_channels,
                               model->config.input_height, model->config.input_width);
    
    cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
    
    float* output;
    cudaMalloc(&output, model->config.batch_size * model->config.num_filters * 
               model->config.input_height * model->config.input_width * sizeof(float));
    
    // Input convolution
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, batch_boards,
                            model->input_conv_filter, model->input_conv_weights,
                            conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                            model->workspace, model->workspace_size, &beta, output_descriptor, output);
    
    // Batch normalization
    cudnnBatchNormalizationForwardTraining(cudnn, CUDNN_BATCHNORM_SPATIAL,
                                           &alpha, &beta, input_descriptor, output,
                                           output_descriptor, output,
                                           bn_descriptor, model->input_bn_scale, model->input_bn_bias,
                                           1.0, model->input_bn_mean, model->input_bn_var,
                                           CUDNN_BN_MIN_EPSILON, model->input_bn_mean, model->input_bn_var);
    
    // ReLU activation
    cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, output,
                           &beta, output_descriptor, output);
    
    // Residual blocks
    float* prev_output = output;
    for (int i = 0; i < model->config.num_residual_blocks; i++) {
        float* res_output;
        cudaMalloc(&res_output, model->config.batch_size * model->config.num_filters * 
                   model->config.input_height * model->config.input_width * sizeof(float));
        
        // First convolution in residual block
        cudnnConvolutionForward(cudnn, &alpha, input_descriptor, prev_output,
                                model->res_conv_filters[i*2], model->res_conv_weights[i*2],
                                conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                model->workspace, model->workspace_size, &beta, output_descriptor, res_output);
        
        cudnnBatchNormalizationForwardTraining(cudnn, CUDNN_BATCHNORM_SPATIAL,
                                               &alpha, &beta, input_descriptor, res_output,
                                               output_descriptor, res_output,
                                               bn_descriptor, model->res_bn_scales[i*2], model->res_bn_biases[i*2],
                                               1.0, model->res_bn_means[i*2], model->res_bn_vars[i*2],
                                               CUDNN_BN_MIN_EPSILON, model->res_bn_means[i*2], model->res_bn_vars[i*2]);
        
        cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, res_output,
                               &beta, output_descriptor, res_output);
        
        // Second convolution in residual block
        cudnnConvolutionForward(cudnn, &alpha, input_descriptor, res_output,
                                model->res_conv_filters[i*2+1], model->res_conv_weights[i*2+1],
                                conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                model->workspace, model->workspace_size, &beta, output_descriptor, res_output);
        
        cudnnBatchNormalizationForwardTraining(cudnn, CUDNN_BATCHNORM_SPATIAL,
                                               &alpha, &beta, input_descriptor, res_output,
                                               output_descriptor, res_output,
                                               bn_descriptor, model->res_bn_scales[i*2+1], model->res_bn_biases[i*2+1],
                                               1.0, model->res_bn_means[i*2+1], model->res_bn_vars[i*2+1],
                                               CUDNN_BN_MIN_EPSILON, model->res_bn_means[i*2+1], model->res_bn_vars[i*2+1]);
        
        // Add residual connection
        cudaMemcpy(output, prev_output, model->config.batch_size * model->config.num_filters * 
                   model->config.input_height * model->config.input_width * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(output, res_output, model->config.batch_size * model->config.num_filters * 
                   model->config.input_height * model->config.input_width * sizeof(float), cudaMemcpyDeviceToDevice);
        
        cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, output,
                               &beta, output_descriptor, output);
        
        cudaFree(res_output);
        prev_output = output;
    }
    
    // Policy head
    float* policy_output;
    cudaMalloc(&policy_output, model->config.batch_size * 2 * model->config.input_height * model->config.input_width * sizeof(float));
    
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, output,
                            model->policy_conv_filter, model->policy_conv_weights,
                            conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                            model->workspace, model->workspace_size, &beta, output_descriptor, policy_output);
    
    cudnnBatchNormalizationForwardTraining(cudnn, CUDNN_BATCHNORM_SPATIAL,
                                           &alpha, &beta, input_descriptor, policy_output,
                                           output_descriptor, policy_output,
                                           bn_descriptor, model->policy_bn_scale, model->policy_bn_bias,
                                           1.0, model->policy_bn_mean, model->policy_bn_var,
                                           CUDNN_BN_MIN_EPSILON, model->policy_bn_mean, model->policy_bn_var);
    
    // Use PyTorch for the fully connected layer and softmax
    auto policy_tensor = torch::from_blob(policy_output, {model->config.batch_size, 2, model->config.input_height, model->config.input_width}, torch::kCUDA);
    auto policy_fc_weight = torch::from_blob(model->policy_fc_weights, {model->config.num_actions, 2 * model->config.input_height * model->config.input_width}, torch::kCUDA);
    auto policy_fc_bias = torch::from_blob(model->policy_fc_bias, {model->config.num_actions}, torch::kCUDA);
    
    policy_tensor = torch::nn::functional::linear(policy_tensor.view({model->config.batch_size, -1}), policy_fc_weight, policy_fc_bias);
    policy_tensor = torch::nn::functional::softmax(policy_tensor, /*dim=*/1);
    
    // Value head
    float* value_output;
    cudaMalloc(&value_output, model->config.batch_size * model->config.input_height * model->config.input_width * sizeof(float));
    
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, output,
                            model->value_conv_filter, model->value_conv_weights,
                            conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                            model->workspace, model->workspace_size, &beta, output_descriptor, value_output);
    
    cudnnBatchNormalizationForwardTraining(cudnn, CUDNN_BATCHNORM_SPATIAL,
                                           &alpha, &beta, input_descriptor, value_output,
                                           output_descriptor, value_output,
                                           bn_descriptor, model->value_bn_scale, model->value_bn_bias,
                                           1.0, model->value_bn_mean, model->value_bn_var,
                                           CUDNN_BN_MIN_EPSILON, model->value_bn_mean, model->value_bn_var);
    
    auto value_tensor = torch::from_blob(value_output, {model->config.batch_size, 1, model->config.input_height, model->config.input_width}, torch::kCUDA);
    auto value_fc1_weight = torch::from_blob(model->value_fc1_weights, {256, model->config.input_height * model->config.input_width}, torch::kCUDA);
    auto value_fc1_bias = torch::from_blob(model->value_fc1_bias, {256}, torch::kCUDA);
    auto value_fc2_weight = torch::from_blob(model->value_fc2_weights, {1, 256}, torch::kCUDA);
    auto value_fc2_bias = torch::from_blob(model->value_fc2_bias, {1}, torch::kCUDA);
    
    value_tensor = torch::nn::functional::linear(value_tensor.view({model->config.batch_size, -1}), value_fc1_weight, value_fc1_bias);
    value_tensor = torch::relu(value_tensor);
    value_tensor = torch::nn::functional::linear(value_tensor, value_fc2_weight, value_fc2_bias);
    value_tensor = torch::tanh(value_tensor);
    
    // Copy results back to output pointers
    cudaMalloc(out_pi, model->config.batch_size * model->config.num_actions * sizeof(float));
    cudaMalloc(out_v, model->config.batch_size * sizeof(float));
    cudaMemcpy(*out_pi, policy_tensor.data_ptr(), model->config.batch_size * model->config.num_actions * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(*out_v, value_tensor.data_ptr(), model->config.batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Cleanup
    cudaFree(output);
    cudaFree(policy_output);
    cudaFree(value_output);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudnnDestroyTensorDescriptor(bn_descriptor);
}

__global__ void policy_loss_kernel(float* target_pi, float* out_pi, float* loss, int batch_size, int action_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sample_loss = 0.0f;
        for (int a = 0; a < action_size; a++) {
            int i = idx * action_size + a;
            sample_loss -= target_pi[i] * logf(out_pi[i] + 1e-8f);
        }
        loss[idx] = sample_loss;
    }
}

__global__ void value_loss_kernel(float* target_v, float* out_v, float* loss, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float diff = target_v[idx] - out_v[idx];
        loss[idx] = diff * diff;
    }
}

std::pair<float, float> compute_losses(float* target_pi, float* target_v, float* out_pi, float* out_v, int batch_size, int action_size) {
    auto target_pi_tensor = torch::from_blob(target_pi, {batch_size, action_size}, torch::kCUDA);
    auto target_v_tensor = torch::from_blob(target_v, {batch_size}, torch::kCUDA);
    auto out_pi_tensor = torch::from_blob(out_pi, {batch_size, action_size}, torch::kCUDA);
    auto out_v_tensor = torch::from_blob(out_v, {batch_size}, torch::kCUDA);

    auto pi_loss = torch::nn::functional::kl_div(out_pi_tensor.log(), target_pi_tensor, torch::kSum);
    auto v_loss = torch::mse_loss(out_v_tensor, target_v_tensor, torch::kSum);

    return {pi_loss.item<float>() / batch_size, v_loss.item<float>() / batch_size};
}

void backward_resnet(ResNetModel* model, float* batch_boards, float* target_pi, float* target_v, float* out_pi, float* out_v) {
    cudnnHandle_t cudnn = model->cudnn_handle;
    float alpha = 1.0f, beta = 0.0f;

    // Convert inputs to PyTorch tensors
    auto boards_tensor = torch::from_blob(batch_boards, {model->config.batch_size, model->config.input_channels, model->config.input_height, model->config.input_width}, torch::kCUDA).requires_grad_();
    auto target_pi_tensor = torch::from_blob(target_pi, {model->config.batch_size, model->config.num_actions}, torch::kCUDA);
    auto target_v_tensor = torch::from_blob(target_v, {model->config.batch_size}, torch::kCUDA);
    auto out_pi_tensor = torch::from_blob(out_pi, {model->config.batch_size, model->config.num_actions}, torch::kCUDA).requires_grad_();
    auto out_v_tensor = torch::from_blob(out_v, {model->config.batch_size}, torch::kCUDA).requires_grad_();

    // Compute losses
    auto pi_loss = torch::nn::functional::kl_div(out_pi_tensor.log(), target_pi_tensor);
    auto v_loss = torch::mse_loss(out_v_tensor, target_v_tensor);
    auto total_loss = pi_loss + v_loss;

    // Backward pass
    total_loss.backward();

    // Get gradients
    auto d_out_pi = out_pi_tensor.grad().contiguous();
    auto d_out_v = out_v_tensor.grad().contiguous();

    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t conv_descriptor;

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnCreateConvolutionDescriptor(&conv_descriptor);

    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, model->config.num_filters,
                               model->config.input_height, model->config.input_width);

    // Policy head backward
    float* d_policy_conv;
    cudaMalloc(&d_policy_conv, model->config.batch_size * 2 * model->config.input_height * model->config.input_width * sizeof(float));

    // FC layer backward (using PyTorch)
    auto d_policy_fc = torch::nn::functional::linear(d_out_pi, 
                                                     torch::from_blob(model->policy_fc_weights, {model->config.num_actions, 2 * model->config.input_height * model->config.input_width}, torch::kCUDA).t());
    cudaMemcpy(d_policy_conv, d_policy_fc.data_ptr(), model->config.batch_size * 2 * model->config.input_height * model->config.input_width * sizeof(float), cudaMemcpyDeviceToDevice);

    // Policy convolution backward
    cudnnConvolutionBackwardData(cudnn, &alpha, model->policy_conv_filter, model->policy_conv_weights,
                                 output_descriptor, d_policy_conv, conv_descriptor,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, model->workspace, model->workspace_size,
                                 &beta, input_descriptor, model->d_policy_conv_weights);

    cudnnConvolutionBackwardFilter(cudnn, &alpha, input_descriptor, batch_boards,
                                   output_descriptor, d_policy_conv, conv_descriptor,
                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, model->workspace, model->workspace_size,
                                   &beta, model->policy_conv_filter, model->d_policy_conv_weights);

    // Value head backward
    float* d_value_conv;
    cudaMalloc(&d_value_conv, model->config.batch_size * model->config.input_height * model->config.input_width * sizeof(float));

    // FC layers backward (using PyTorch)
    auto d_value_fc2 = torch::nn::functional::linear(d_out_v, 
                                                     torch::from_blob(model->value_fc2_weights, {1, 256}, torch::kCUDA).t());
    auto d_value_fc1 = torch::nn::functional::linear(torch::relu(d_value_fc2), 
                                                     torch::from_blob(model->value_fc1_weights, {256, model->config.input_height * model->config.input_width}, torch::kCUDA).t());
    cudaMemcpy(d_value_conv, d_value_fc1.data_ptr(), model->config.batch_size * model->config.input_height * model->config.input_width * sizeof(float), cudaMemcpyDeviceToDevice);

    // Value convolution backward
    cudnnConvolutionBackwardData(cudnn, &alpha, model->value_conv_filter, model->value_conv_weights,
                                 output_descriptor, d_value_conv, conv_descriptor,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, model->workspace, model->workspace_size,
                                 &beta, input_descriptor, model->d_value_conv_weights);

    cudnnConvolutionBackwardFilter(cudnn, &alpha, input_descriptor, batch_boards,
                                   output_descriptor, d_value_conv, conv_descriptor,
                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, model->workspace, model->workspace_size,
                                   &beta, model->value_conv_filter, model->d_value_conv_weights);

    // Combine gradients from policy and value heads
    float* d_res_output;
    cudaMalloc(&d_res_output, model->config.batch_size * model->config.num_filters * model->config.input_height * model->config.input_width * sizeof(float));
    cudaMemcpy(d_res_output, d_policy_conv, model->config.batch_size * model->config.num_filters * model->config.input_height * model->config.input_width * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_res_output, d_value_conv, model->config.batch_size * model->config.num_filters * model->config.input_height * model->config.input_width * sizeof(float), cudaMemcpyDeviceToDevice);

    // Residual blocks backward
    for (int i = model->config.num_residual_blocks - 1; i >= 0; i--) {
        float* d_res_input;
        cudaMalloc(&d_res_input, model->config.batch_size * model->config.num_filters * model->config.input_height * model->config.input_width * sizeof(float));

        // Second convolution in residual block
        cudnnConvolutionBackwardData(cudnn, &alpha, model->res_conv_filters[i*2+1], model->res_conv_weights[i*2+1],
                                     output_descriptor, d_res_output, conv_descriptor,
                                     CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, model->workspace, model->workspace_size,
                                     &beta, input_descriptor, d_res_input);

        cudnnConvolutionBackwardFilter(cudnn, &alpha, input_descriptor, batch_boards,
                                       output_descriptor, d_res_output, conv_descriptor,
                                       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, model->workspace, model->workspace_size,
                                       &beta, model->res_conv_filters[i*2+1], model->d_res_conv_weights[i*2+1]);

        // First convolution in residual block
        cudnnConvolutionBackwardData(cudnn, &alpha, model->res_conv_filters[i*2], model->res_conv_weights[i*2],
                                     output_descriptor, d_res_input, conv_descriptor,
                                     CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, model->workspace, model->workspace_size,
                                     &beta, input_descriptor, d_res_output);

        cudnnConvolutionBackwardFilter(cudnn, &alpha, input_descriptor, batch_boards,
                                       output_descriptor, d_res_input, conv_descriptor,
                                       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, model->workspace, model->workspace_size,
                                       &beta, model->res_conv_filters[i*2], model->d_res_conv_weights[i*2]);

        // Add gradient from residual connection
        cudaMemcpy(d_res_output, d_res_input, model->config.batch_size * model->config.num_filters * model->config.input_height * model->config.input_width * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(d_res_input);
    }

    // Input convolution backward
    cudnnConvolutionBackwardData(cudnn, &alpha, model->input_conv_filter, model->input_conv_weights,
                                 output_descriptor, d_res_output, conv_descriptor,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, model->workspace, model->workspace_size,
                                 &beta, input_descriptor, model->d_input_conv_weights);

    cudnnConvolutionBackwardFilter(cudnn, &alpha, input_descriptor, batch_boards,
                                   output_descriptor, d_res_output, conv_descriptor,
                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, model->workspace, model->workspace_size,
                                   &beta, model->input_conv_filter, model->d_input_conv_weights);

    // Cleanup
    cudaFree(d_policy_conv);
    cudaFree(d_value_conv);
    cudaFree(d_res_output);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
}

void adam_update(torch::optim::Adam& optimizer) {
    optimizer.step();
    optimizer.zero_grad();
}