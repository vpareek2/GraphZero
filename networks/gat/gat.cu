#include "gat.cuh"

#include <curand.h>
#include <chrono>
#include <torch/serialize.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <nccl.h>
#include <mpi.h>

/*************************************************************************************************************************************************************
 * CUDA ERROR CHECKING
**************************************************************************************************************************************************************/

#define CUDNN_CHECK(call) { cudnnStatus_t status = call; if (status != CUDNN_STATUS_SUCCESS) { fprintf(stderr, "CUDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); exit(1); } }
#define CUDA_CHECK(call) { cudaError_t status = call; if (status != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); exit(1); } }
#define CURAND_CHECK(call) { curandStatus_t status = call; if (status != CURAND_STATUS_SUCCESS) { fprintf(stderr, "CURAND error at %s:%d: %d\n", __FILE__, __LINE__, status); exit(1); } }
#define NCCL_CHECK(call) { \
    ncclResult_t status = call; \
    if (status != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(status)); \
        exit(1); \
    } \
}
/*************************************************************************************************************************************************************/

static void gat_init(INeuralNet* self, const IGame* game) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Initialize model configuration
    init_model_config(model, game);

    // Initialize cuDNN
    CUDNN_CHECK(cudnnCreate(&model->cudnn_handle));

    // Initialize input block
    CUDA_CHECK(init_input_block(model));

    // Initialize GAT layers
    CUDA_CHECK(init_gat_layers(model));

    // Initialize output block
    CUDA_CHECK(init_output_block(model));

    // Initialize weights
    CUDA_CHECK(init_weights(model));

    // Initialize PyTorch optimizer
    std::vector<torch::Tensor> params;

    // Add all weights and biases to params vector
    // Input block
    params.push_back(torch::from_blob(model->input_weights, {model->config.input_features, model->config.hidden_features}, torch::kCUDA));
    params.push_back(torch::from_blob(model->input_bias, {model->config.hidden_features}, torch::kCUDA));

    // GAT layers
    for (int i = 0; i < model->config.num_layers; i++) {
        params.push_back(torch::from_blob(model->layer_weights[i], {model->config.hidden_features, model->config.hidden_features}, torch::kCUDA));
        params.push_back(torch::from_blob(model->layer_biases[i], {model->config.hidden_features}, torch::kCUDA));
        params.push_back(torch::from_blob(model->attention_weights[i], {model->config.num_heads, 2, model->config.hidden_features}, torch::kCUDA));
    }

    // Output block
    params.push_back(torch::from_blob(model->value_weights, {model->config.hidden_features}, torch::kCUDA));
    params.push_back(torch::from_blob(model->value_bias, {1}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_weights, {model->config.hidden_features, model->config.num_actions}, torch::kCUDA));
    params.push_back(torch::from_blob(model->policy_bias, {model->config.num_actions}, torch::kCUDA));

    model->optimizer = new torch::optim::Adam(params, torch::optim::AdamOptions(model->config.learning_rate).weight_decay(model->config.weight_decay));

    // Allocate workspace for cuDNN
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(model->cudnn_handle, model->input_descriptor, model->layer_descriptors[0],
        /* convolution descriptor */, model->layer_descriptors[0],
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &workspace_size));
    CUDA_CHECK(cudaMalloc(&model->workspace, workspace_size));
    model->workspace_size = workspace_size;
}

static void gat_train(INeuralNet* self, TrainingExample* examples, int num_examples) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Allocate memory for batch data
    float *batch_boards, *batch_pis, *batch_vs;
    CUDA_CHECK(cudaMalloc(&batch_boards, model->config.batch_size * MAX_NODES * MAX_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch_pis, model->config.batch_size * model->config.num_actions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch_vs, model->config.batch_size * sizeof(float)));

    // Allocate memory for output
    float *out_pi, *out_v;
    CUDA_CHECK(cudaMalloc(&out_pi, model->config.batch_size * model->config.num_actions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_v, model->config.batch_size * sizeof(float)));

    for (int epoch = 0; epoch < model->config.epochs; epoch++) {
        printf("EPOCH ::: %d\n", epoch + 1);

        float pi_loss_sum = 0.0f;
        float v_loss_sum = 0.0f;
        int batch_count = num_examples / model->config.batch_size;

        for (int batch = 0; batch < batch_count; batch++) {
            // Prepare batch data
            prepare_batch(examples, num_examples, model->config.batch_size, batch_boards, batch_pis, batch_vs);

            // Forward pass
            auto start_time = std::chrono::high_resolution_clock::now();
            forward_gat(model, batch_boards, &out_pi, &out_v);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            printf("Forward pass took %lld microseconds\n", duration.count());

            // Compute losses
            auto [pi_loss, v_loss] = compute_losses(batch_pis, batch_vs, out_pi, out_v, 
                                                    model->config.batch_size, model->config.num_actions);
            pi_loss_sum += pi_loss;
            v_loss_sum += v_loss;

            // Backward pass
            start_time = std::chrono::high_resolution_clock::now();
            backward_gat(model, batch_boards, batch_pis, batch_vs, out_pi, out_v);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            printf("Backward pass took %lld microseconds\n", duration.count());

            // Update weights using Adam optimizer
            start_time = std::chrono::high_resolution_clock::now();
            model->optimizer->step();
            model->optimizer->zero_grad();
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            printf("Optimizer step took %lld microseconds\n", duration.count());
        }

        // Print epoch results
        printf("Average Policy Loss: %f, Average Value Loss: %f\n", 
               pi_loss_sum / batch_count, v_loss_sum / batch_count);
    }

    // Clean up
    CUDA_CHECK(cudaFree(batch_boards));
    CUDA_CHECK(cudaFree(batch_pis));
    CUDA_CHECK(cudaFree(batch_vs));
    CUDA_CHECK(cudaFree(out_pi));
    CUDA_CHECK(cudaFree(out_v));
}

static void gat_predict(INeuralNet* self, const float* board, float* pi, float* v) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Allocate device memory for input and output
    float *d_board, *d_pi, *d_v;
    CUDA_CHECK(cudaMalloc(&d_board, sizeof(float) * model->config.input_features * model->config.board_size));
    CUDA_CHECK(cudaMalloc(&d_pi, sizeof(float) * model->config.num_actions));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(d_board, board, sizeof(float) * model->config.input_features * model->config.board_size, cudaMemcpyHostToDevice));

    // Forward pass
    forward_gat(model, d_board, &d_pi, &d_v);

    // Apply softmax to policy output (on GPU)
    dim3 grid_policy((model->config.num_actions + 255) / 256);
    dim3 block_policy(256);
    softmax<<<grid_policy, block_policy>>>(d_pi, model->config.num_actions);
    CUDA_CHECK(cudaGetLastError());

    // Apply tanh to value output (on GPU)
    tanh_activate<<<1, 1>>>(d_v, 1);
    CUDA_CHECK(cudaGetLastError());

    // Copy output back to host
    CUDA_CHECK(cudaMemcpyAsync(pi, d_pi, sizeof(float) * model->config.num_actions, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(v, d_v, sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_board));
    CUDA_CHECK(cudaFree(d_pi));
    CUDA_CHECK(cudaFree(d_v));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Prediction time: %f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}


static void gat_save_checkpoint(INeuralNet* self, const char* folder, const char* filename) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    char filepath[MAX_FILENAME_LENGTH];
    snprintf(filepath, MAX_FILENAME_LENGTH, "%s/%s", folder, filename);

    torch::serialize::OutputArchive archive;

    // Save model configuration
    archive.write("config", torch::from_blob(&model->config, {sizeof(ModelConfig)}, torch::kByte));

    // Save input block weights
    archive.write("input_weights", torch::from_blob(model->input_weights, 
        {model->config.input_features, model->config.hidden_features}, torch::kFloat32));
    archive.write("input_bias", torch::from_blob(model->input_bias, 
        {model->config.hidden_features}, torch::kFloat32));

    // Save GAT layer weights
    for (int i = 0; i < model->config.num_layers; i++) {
        char key[50];
        snprintf(key, sizeof(key), "layer_weights_%d", i);
        archive.write(key, torch::from_blob(model->layer_weights[i], 
            {model->config.hidden_features, model->config.hidden_features}, torch::kFloat32));
        
        snprintf(key, sizeof(key), "layer_biases_%d", i);
        archive.write(key, torch::from_blob(model->layer_biases[i], 
            {model->config.hidden_features}, torch::kFloat32));
        
        snprintf(key, sizeof(key), "attention_weights_%d", i);
        archive.write(key, torch::from_blob(model->attention_weights[i], 
            {model->config.num_heads, 2, model->config.hidden_features}, torch::kFloat32));
    }

    // Save output block weights
    archive.write("value_weights", torch::from_blob(model->value_weights, 
        {model->config.hidden_features}, torch::kFloat32));
    archive.write("value_bias", torch::from_blob(model->value_bias, {1}, torch::kFloat32));
    archive.write("policy_weights", torch::from_blob(model->policy_weights, 
        {model->config.hidden_features, model->config.num_actions}, torch::kFloat32));
    archive.write("policy_bias", torch::from_blob(model->policy_bias, 
        {model->config.num_actions}, torch::kFloat32));

    // Save optimizer state
    archive.write("optimizer", model->optimizer->state_dict());

    try {
        torch::serialize::save_to_file(archive, filepath);
    } catch (const c10::Error& e) {
        fprintf(stderr, "Error saving checkpoint: %s\n", e.what());
    }
}

static void gat_load_checkpoint(INeuralNet* self, const char* folder, const char* filename) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    char filepath[MAX_FILENAME_LENGTH];
    snprintf(filepath, MAX_FILENAME_LENGTH, "%s/%s", folder, filename);

    torch::serialize::InputArchive archive;
    try {
        torch::serialize::load_from_file(archive, filepath);
    } catch (const c10::Error& e) {
        fprintf(stderr, "Error loading checkpoint: %s\n", e.what());
        return;
    }

    // Load model configuration
    torch::Tensor config_tensor;
    archive.read("config", config_tensor);
    memcpy(&model->config, config_tensor.data_ptr(), sizeof(ModelConfig));

    // Load input block weights
    torch::Tensor input_weights, input_bias;
    archive.read("input_weights", input_weights);
    archive.read("input_bias", input_bias);
    CUDA_CHECK(cudaMemcpy(model->input_weights, input_weights.data_ptr(), 
        input_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(model->input_bias, input_bias.data_ptr(), 
        input_bias.numel() * sizeof(float), cudaMemcpyDeviceToDevice));

    // Load GAT layer weights
    for (int i = 0; i < model->config.num_layers; i++) {
        char key[50];
        torch::Tensor layer_weights, layer_biases, attention_weights;
        
        snprintf(key, sizeof(key), "layer_weights_%d", i);
        archive.read(key, layer_weights);
        CUDA_CHECK(cudaMemcpy(model->layer_weights[i], layer_weights.data_ptr(), 
            layer_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
        
        snprintf(key, sizeof(key), "layer_biases_%d", i);
        archive.read(key, layer_biases);
        CUDA_CHECK(cudaMemcpy(model->layer_biases[i], layer_biases.data_ptr(), 
            layer_biases.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
        
        snprintf(key, sizeof(key), "attention_weights_%d", i);
        archive.read(key, attention_weights);
        CUDA_CHECK(cudaMemcpy(model->attention_weights[i], attention_weights.data_ptr(), 
            attention_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Load output block weights
    torch::Tensor value_weights, value_bias, policy_weights, policy_bias;
    archive.read("value_weights", value_weights);
    archive.read("value_bias", value_bias);
    archive.read("policy_weights", policy_weights);
    archive.read("policy_bias", policy_bias);
    CUDA_CHECK(cudaMemcpy(model->value_weights, value_weights.data_ptr(), 
        value_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(model->value_bias, value_bias.data_ptr(), 
        value_bias.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(model->policy_weights, policy_weights.data_ptr(), 
        policy_weights.numel() * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(model->policy_bias, policy_bias.data_ptr(), 
        policy_bias.numel() * sizeof(float), cudaMemcpyDeviceToDevice));

    // Load optimizer state
    torch::serialize::OutputArchive optimizer_archive;
    archive.read("optimizer", optimizer_archive);
    model->optimizer->load_state_dict(optimizer_archive);

    // Recreate cuDNN descriptors
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, model->config.max_nodes, model->config.input_features));

    for (int i = 0; i < model->config.num_layers; i++) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->layer_descriptors[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   model->config.batch_size, model->config.num_heads, model->config.max_nodes, model->config.hidden_features));
    }

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->value_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.hidden_features));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->policy_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.num_actions));
}

static void gat_destroy(INeuralNet* self) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Free input block resources
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(model->input_descriptor));
    CUDA_CHECK(cudaFree(model->input_weights));
    CUDA_CHECK(cudaFree(model->input_bias));

    // Free GAT layer resources
    for (int i = 0; i < model->config.num_layers; i++) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(model->layer_descriptors[i]));
        CUDA_CHECK(cudaFree(model->layer_weights[i]));
        CUDA_CHECK(cudaFree(model->layer_biases[i]));
        CUDA_CHECK(cudaFree(model->attention_weights[i]));
    }
    free(model->layer_descriptors);
    free(model->layer_weights);
    free(model->layer_biases);
    free(model->attention_weights);

    // Free output block resources
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(model->value_descriptor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(model->policy_descriptor));
    CUDA_CHECK(cudaFree(model->value_weights));
    CUDA_CHECK(cudaFree(model->value_bias));
    CUDA_CHECK(cudaFree(model->policy_weights));
    CUDA_CHECK(cudaFree(model->policy_bias));

    // Free cuDNN workspace
    CUDA_CHECK(cudaFree(model->workspace));

    // Destroy cuDNN handle
    CUDNN_CHECK(cudnnDestroy(model->cudnn_handle));

    // Delete PyTorch optimizer
    delete model->optimizer;

    // Free the wrapper itself
    free(wrapper);
}

static void gat_train_distributed(INeuralNet* self, float* d_boards, float* d_pis, float* d_vs, int num_examples, int world_rank, int world_size, ncclComm_t nccl_comm, cudaStream_t cuda_stream) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Efficient data distribution
    int examples_per_gpu = (num_examples + world_size - 1) / world_size;
    int local_examples = min(examples_per_gpu, num_examples - world_rank * examples_per_gpu);

    // Broadcast initial data to all GPUs
    NCCL_CHECK(ncclBroadcast(d_boards, d_boards, num_examples * model->config.input_features * sizeof(float), ncclFloat32, 0, nccl_comm, cuda_stream));
    NCCL_CHECK(ncclBroadcast(d_pis, d_pis, num_examples * model->config.num_actions * sizeof(float), ncclFloat32, 0, nccl_comm, cuda_stream));
    NCCL_CHECK(ncclBroadcast(d_vs, d_vs, num_examples * sizeof(float), ncclFloat32, 0, nccl_comm, cuda_stream));

    // Model parallelism: Divide layers across GPUs
    int layers_per_gpu = (model->config.num_layers + world_size - 1) / world_size;
    int start_layer = world_rank * layers_per_gpu;
    int end_layer = min((world_rank + 1) * layers_per_gpu, model->config.num_layers);

    for (int epoch = 0; epoch < model->config.epochs; epoch++) {
        float pi_loss_sum = 0.0f;
        float v_loss_sum = 0.0f;
        int batch_count = (local_examples + model->config.batch_size - 1) / model->config.batch_size;

        for (int batch = 0; batch < batch_count; batch++) {
            int batch_start = batch * model->config.batch_size;
            int batch_size = min(model->config.batch_size, local_examples - batch_start);

            // Allocate memory for layer inputs and outputs
            float *layer_input, *layer_output, *out_pi, *out_v;
            CUDA_CHECK(cudaMalloc(&layer_input, batch_size * model->config.hidden_features * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&layer_output, batch_size * model->config.hidden_features * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&out_pi, batch_size * model->config.num_actions * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&out_v, batch_size * sizeof(float)));

            // Copy input data to layer_input for the first GPU
            if (world_rank == 0) {
                CUDA_CHECK(cudaMemcpy(layer_input, d_boards + batch_start * model->config.input_features,
                                      batch_size * model->config.input_features * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
            }

            // Forward pass with model parallelism
            for (int layer = start_layer; layer < end_layer; layer++) {
                forward_gat_layer(model, layer, layer_input, layer_output, batch_size);
                
                // Communicate layer output to next GPU
                if (layer < model->config.num_layers - 1) {
                    int next_rank = (world_rank + 1) % world_size;
                    NCCL_CHECK(ncclSend(layer_output, batch_size * model->config.hidden_features, ncclFloat32, next_rank, nccl_comm, cuda_stream));
                    if (layer < end_layer - 1) {
                        NCCL_CHECK(ncclRecv(layer_input, batch_size * model->config.hidden_features, ncclFloat32, (world_rank - 1 + world_size) % world_size, nccl_comm, cuda_stream));
                    }
                }
                // Swap layer_input and layer_output pointers
                float* temp = layer_input;
                layer_input = layer_output;
                layer_output = temp;
            }

            // Final output layer
            if (world_rank == world_size - 1) {
                compute_output(model, layer_input, out_pi, out_v, batch_size);
            }

            // Broadcast output to all GPUs
            NCCL_CHECK(ncclBroadcast(out_pi, out_pi, batch_size * model->config.num_actions, ncclFloat32, world_size - 1, nccl_comm, cuda_stream));
            NCCL_CHECK(ncclBroadcast(out_v, out_v, batch_size, ncclFloat32, world_size - 1, nccl_comm, cuda_stream));

            // Compute losses
            auto [pi_loss, v_loss] = compute_losses(d_pis + batch_start * model->config.num_actions, 
                                                    d_vs + batch_start, 
                                                    out_pi, out_v, 
                                                    batch_size, model->config.num_actions);
            pi_loss_sum += pi_loss;
            v_loss_sum += v_loss;

            // Backward pass with model parallelism
            float* grad_output;
            CUDA_CHECK(cudaMalloc(&grad_output, batch_size * model->config.hidden_features * sizeof(float)));

            // Initialize grad_output for the last layer
            if (world_rank == world_size - 1) {
                // Compute gradients for policy and value heads
                float *d_policy, *d_value;
                CUDA_CHECK(cudaMalloc(&d_policy, batch_size * model->config.num_actions * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_value, batch_size * sizeof(float)));

                softmax_cross_entropy_gradient<<<(model->config.num_actions + 255) / 256, 256>>>(
                    out_pi, d_pis + batch_start * model->config.num_actions, d_policy, batch_size, model->config.num_actions);
                mse_gradient<<<(batch_size + 255) / 256, 256>>>(
                    out_v, d_vs + batch_start, d_value, batch_size);

                // Compute initial grad_output
                compute_initial_grad_output(model, d_policy, d_value, grad_output, batch_size);

                CUDA_CHECK(cudaFree(d_policy));
                CUDA_CHECK(cudaFree(d_value));
            }

            for (int layer = end_layer - 1; layer >= start_layer; layer--) {
                backward_gat_layer(model, layer, layer_input, layer_output, grad_output, batch_size);

                // Communicate gradients to previous GPU
                if (layer > 0) {
                    int prev_rank = (world_rank - 1 + world_size) % world_size;
                    NCCL_CHECK(ncclSend(grad_output, batch_size * model->config.hidden_features, ncclFloat32, prev_rank, nccl_comm, cuda_stream));
                    if (layer > start_layer) {
                        NCCL_CHECK(ncclRecv(grad_output, batch_size * model->config.hidden_features, ncclFloat32, (world_rank + 1) % world_size, nccl_comm, cuda_stream));
                    }
                }
                // Swap layer_input and layer_output pointers
                float* temp = layer_input;
                layer_input = layer_output;
                layer_output = temp;
            }

            // Aggregate gradients across GPUs
            for (int i = start_layer; i < end_layer; i++) {
                NCCL_CHECK(ncclAllReduce(model->d_layer_weights[i], model->d_layer_weights[i], 
                                         model->config.hidden_features * model->config.hidden_features, 
                                         ncclFloat32, ncclSum, nccl_comm, cuda_stream));
                NCCL_CHECK(ncclAllReduce(model->d_attention_weights[i], model->d_attention_weights[i], 
                                         model->config.num_heads * 2 * model->config.hidden_features, 
                                         ncclFloat32, ncclSum, nccl_comm, cuda_stream));
            }

            if (world_rank == world_size - 1) {
                NCCL_CHECK(ncclAllReduce(model->d_value_weights, model->d_value_weights, 
                                         model->config.hidden_features, ncclFloat32, ncclSum, 
                                         nccl_comm, cuda_stream));
                NCCL_CHECK(ncclAllReduce(model->d_policy_weights, model->d_policy_weights, 
                                         model->config.hidden_features * model->config.num_actions, 
                                         ncclFloat32, ncclSum, nccl_comm, cuda_stream));
            }

            // Update weights using Adam optimizer
            model->optimizer->step();
            model->optimizer->zero_grad();

            // Clean up
            CUDA_CHECK(cudaFree(layer_input));
            CUDA_CHECK(cudaFree(layer_output));
            CUDA_CHECK(cudaFree(out_pi));
            CUDA_CHECK(cudaFree(out_v));
            CUDA_CHECK(cudaFree(grad_output));
        }

        // Synchronize CUDA stream
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

        // Compute global average losses
        float global_pi_loss, global_v_loss;
        MPI_Allreduce(&pi_loss_sum, &global_pi_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&v_loss_sum, &global_v_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        global_pi_loss /= (num_examples * world_size);
        global_v_loss /= (num_examples * world_size);

        if (world_rank == 0) {
            printf("Epoch %d, Average Policy Loss: %f, Average Value Loss: %f\n", 
                   epoch + 1, global_pi_loss, global_v_loss);
        }
    }
}

static void gat_broadcast_weights(INeuralNet* self, int world_rank, int world_size, ncclComm_t nccl_comm, cudaStream_t cuda_stream) {
    GATWrapper* wrapper = (GATWrapper*)self;
    GATModel* model = &wrapper->model;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, cuda_stream));

    // Calculate total size of all weights
    size_t total_weight_size = 0;
    total_weight_size += model->config.input_features * model->config.hidden_features; // input_weights
    total_weight_size += model->config.hidden_features; // input_bias
    for (int i = 0; i < model->config.num_layers; i++) {
        total_weight_size += model->config.hidden_features * model->config.hidden_features; // layer_weights
        total_weight_size += model->config.hidden_features; // layer_biases
        total_weight_size += model->config.num_heads * 2 * model->config.hidden_features; // attention_weights
    }
    total_weight_size += model->config.hidden_features; // value_weights
    total_weight_size += 1; // value_bias
    total_weight_size += model->config.hidden_features * model->config.num_actions; // policy_weights
    total_weight_size += model->config.num_actions; // policy_bias

    // Allocate a single contiguous buffer for all weights
    float* all_weights;
    CUDA_CHECK(cudaMalloc(&all_weights, total_weight_size * sizeof(float)));

    // Copy all weights into the contiguous buffer
    size_t offset = 0;
    CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->input_weights, 
                               model->config.input_features * model->config.hidden_features * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.input_features * model->config.hidden_features;
    CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->input_bias, 
                               model->config.hidden_features * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.hidden_features;

    for (int i = 0; i < model->config.num_layers; i++) {
        CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->layer_weights[i], 
                                   model->config.hidden_features * model->config.hidden_features * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        offset += model->config.hidden_features * model->config.hidden_features;
        CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->layer_biases[i], 
                                   model->config.hidden_features * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        offset += model->config.hidden_features;
        CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->attention_weights[i], 
                                   model->config.num_heads * 2 * model->config.hidden_features * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        offset += model->config.num_heads * 2 * model->config.hidden_features;
    }

    CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->value_weights, 
                               model->config.hidden_features * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.hidden_features;
    CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->value_bias, 
                               sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += 1;
    CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->policy_weights, 
                               model->config.hidden_features * model->config.num_actions * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.hidden_features * model->config.num_actions;
    CUDA_CHECK(cudaMemcpyAsync(all_weights + offset, model->policy_bias, 
                               model->config.num_actions * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));

    // Perform ring-allreduce operation
    NCCL_CHECK(ncclAllReduce(all_weights, all_weights, total_weight_size, ncclFloat32, ncclSum, nccl_comm, cuda_stream));

    // Divide by world_size to get the average
    float scale = 1.0f / world_size;
    CUDA_CHECK(cublasSscal(model->cublas_handle, total_weight_size, &scale, all_weights, 1));

    // Copy the synchronized weights back to their original locations
    offset = 0;
    CUDA_CHECK(cudaMemcpyAsync(model->input_weights, all_weights + offset, 
                               model->config.input_features * model->config.hidden_features * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.input_features * model->config.hidden_features;
    CUDA_CHECK(cudaMemcpyAsync(model->input_bias, all_weights + offset, 
                               model->config.hidden_features * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.hidden_features;

    for (int i = 0; i < model->config.num_layers; i++) {
        CUDA_CHECK(cudaMemcpyAsync(model->layer_weights[i], all_weights + offset, 
                                   model->config.hidden_features * model->config.hidden_features * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        offset += model->config.hidden_features * model->config.hidden_features;
        CUDA_CHECK(cudaMemcpyAsync(model->layer_biases[i], all_weights + offset, 
                                   model->config.hidden_features * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        offset += model->config.hidden_features;
        CUDA_CHECK(cudaMemcpyAsync(model->attention_weights[i], all_weights + offset, 
                                   model->config.num_heads * 2 * model->config.hidden_features * sizeof(float), 
                                   cudaMemcpyDeviceToDevice, cuda_stream));
        offset += model->config.num_heads * 2 * model->config.hidden_features;
    }

    CUDA_CHECK(cudaMemcpyAsync(model->value_weights, all_weights + offset, 
                               model->config.hidden_features * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.hidden_features;
    CUDA_CHECK(cudaMemcpyAsync(model->value_bias, all_weights + offset, 
                               sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += 1;
    CUDA_CHECK(cudaMemcpyAsync(model->policy_weights, all_weights + offset, 
                               model->config.hidden_features * model->config.num_actions * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));
    offset += model->config.hidden_features * model->config.num_actions;
    CUDA_CHECK(cudaMemcpyAsync(model->policy_bias, all_weights + offset, 
                               model->config.num_actions * sizeof(float), 
                               cudaMemcpyDeviceToDevice, cuda_stream));

    // Free the temporary buffer
    CUDA_CHECK(cudaFree(all_weights));

    // Record end time and calculate duration
    CUDA_CHECK(cudaEventRecord(stop, cuda_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    if (world_rank == 0) {
        printf("Weight synchronization time: %f ms\n", milliseconds);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

INeuralNet* create_gat_model(const IGame* game) {
    GATWrapper* wrapper = (GATWrapper*)malloc(sizeof(GATWrapper));
    wrapper->base.impl = wrapper;
    wrapper->base.init = gat_init;
    wrapper->base.train = gat_train;
    wrapper->base.predict = gat_predict;
    wrapper->base.save_checkpoint = gat_save_checkpoint;
    wrapper->base.load_checkpoint = gat_load_checkpoint;
    wrapper->base.destroy = gat_destroy;
    wrapper->base.train_distributed = gat_train_distributed;
    wrapper->base.broadcast_weights = gat_broadcast_weights;

    gat_init(&wrapper->base, game);

    return &wrapper->base;
}

/*************************************************************************************************************************************************************
 * INIT HELPER FUNCTIONS
**************************************************************************************************************************************************************/

static void init_model_config(GATModel* model, const IGame* game) {
    int board_size = game->get_board_size(game);
    model->config.input_features = board_size * board_size;  // Assuming square board
    model->config.hidden_features = 256;  // You can adjust this
    model->config.output_features = 256;  // You can adjust this
    model->config.num_heads = 8;  // Typical value, can be adjusted
    model->config.num_layers = 3;  // You can adjust this
    model->config.num_actions = game->get_action_size(game);
    model->config.max_nodes = board_size * board_size;
    model->config.max_edges = model->config.max_nodes * model->config.max_nodes;  // Fully connected graph
    model->config.learning_rate = 0.001f;
    model->config.weight_decay = 0.0001f;
    model->config.dropout = 0.1f;
    model->config.alpha = 0.2f;  // LeakyReLU angle
    model->config.batch_size = 64;  // You can adjust this
    model->config.epochs = 10;  // You can adjust this
}

static void init_input_block(GATModel* model) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->input_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, model->config.max_nodes, model->config.input_features));

    CUDA_CHECK(cudaMalloc(&model->input_weights, sizeof(float) * model->config.input_features * model->config.hidden_features));
    CUDA_CHECK(cudaMalloc(&model->input_bias, sizeof(float) * model->config.hidden_features));
}

static void init_gat_layers(GATModel* model) {
    model->layer_descriptors = (cudnnTensorDescriptor_t*)malloc(model->config.num_layers * sizeof(cudnnTensorDescriptor_t));
    model->layer_weights = (float**)malloc(model->config.num_layers * sizeof(float*));
    model->layer_biases = (float**)malloc(model->config.num_layers * sizeof(float*));
    model->attention_weights = (float**)malloc(model->config.num_layers * sizeof(float*));

    if (!model->layer_descriptors || !model->layer_weights || !model->layer_biases || !model->attention_weights) {
        fprintf(stderr, "Memory allocation failed in init_gat_layers\n");
        exit(1);
    }

    for (int i = 0; i < model->config.num_layers; i++) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->layer_descriptors[i]));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->layer_descriptors[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   model->config.batch_size, model->config.num_heads, model->config.max_nodes, model->config.hidden_features));

        CUDA_CHECK(cudaMalloc(&model->layer_weights[i], sizeof(float) * model->config.hidden_features * model->config.hidden_features));
        CUDA_CHECK(cudaMalloc(&model->layer_biases[i], sizeof(float) * model->config.hidden_features));
        CUDA_CHECK(cudaMalloc(&model->attention_weights[i], sizeof(float) * model->config.num_heads * 2 * model->config.hidden_features));
    }
}

static void init_output_block(GATModel* model) {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->value_descriptor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&model->policy_descriptor));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->value_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.hidden_features));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(model->policy_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               model->config.batch_size, 1, 1, model->config.num_actions));

    CUDA_CHECK(cudaMalloc(&model->value_weights, sizeof(float) * model->config.hidden_features));
    CUDA_CHECK(cudaMalloc(&model->value_bias, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->policy_weights, sizeof(float) * model->config.hidden_features * model->config.num_actions));
    CUDA_CHECK(cudaMalloc(&model->policy_bias, sizeof(float) * model->config.num_actions));
}

static void init_weights(GATModel* model) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    // Initialize input weights
    CURAND_CHECK(curandGenerateNormal(gen, model->input_weights, model->config.input_features * model->config.hidden_features, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->input_bias, model->config.hidden_features, 0, 0.1));

    // Initialize GAT layer weights
    for (int i = 0; i < model->config.num_layers; i++) {
        CURAND_CHECK(curandGenerateNormal(gen, model->layer_weights[i], model->config.hidden_features * model->config.hidden_features, 0, 0.1));
        CURAND_CHECK(curandGenerateNormal(gen, model->layer_biases[i], model->config.hidden_features, 0, 0.1));
        CURAND_CHECK(curandGenerateNormal(gen, model->attention_weights[i], model->config.num_heads * 2 * model->config.hidden_features, 0, 0.1));
    }

    // Initialize output weights
    CURAND_CHECK(curandGenerateNormal(gen, model->value_weights, model->config.hidden_features, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->value_bias, 1, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->policy_weights, model->config.hidden_features * model->config.num_actions, 0, 0.1));
    CURAND_CHECK(curandGenerateNormal(gen, model->policy_bias, model->config.num_actions, 0, 0.1));

    CURAND_CHECK(curandDestroyGenerator(gen));
}

/*************************************************************************************************************************************************************
 * TRAIN HELPER FUNCTIONS
**************************************************************************************************************************************************************/

// Helper function to prepare batch data
static void prepare_batch(TrainingExample* examples, int num_examples, int batch_size,
                          float* batch_boards, float* batch_pis, float* batch_vs) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use cuRAND for random selection
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, batch_size * sizeof(int)));
    CURAND_CHECK(curandGenerate(gen, (unsigned int*)d_indices, batch_size));

    // Custom CUDA kernel to prepare batch
    dim3 grid((batch_size + 255) / 256);
    dim3 block(256);
    prepare_batch_kernel<<<grid, block>>>(examples, num_examples, d_indices, batch_boards, batch_pis, batch_vs, batch_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_indices));
    CURAND_CHECK(curandDestroyGenerator(gen));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Batch preparation took %lld microseconds\n", duration.count());
}

// CUDA kernel for batch preparation
__global__ void prepare_batch_kernel(TrainingExample* examples, int num_examples, int* indices,
                                     float* batch_boards, float* batch_pis, float* batch_vs, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int example_idx = indices[idx] % num_examples;
        memcpy(batch_boards + idx * MAX_NODES * MAX_FEATURES, examples[example_idx].board, MAX_NODES * MAX_FEATURES * sizeof(float));
        memcpy(batch_pis + idx * NUM_ACTIONS, examples[example_idx].pi, NUM_ACTIONS * sizeof(float));
        batch_vs[idx] = examples[example_idx].v;
    }
}

// Forward pass for a single GAT layer
static void forward_gat_layer(GATModel* model, int layer_index, float* layer_input, float* layer_output, int batch_size) {
    cudnnHandle_t cudnn = model->cudnn_handle;
    cublasHandle_t cublas;
    CUDA_CHECK(cublasCreate(&cublas));

    float alpha = 1.0f, beta = 0.0f;

    // Attention mechanism (custom CUDA kernel)
    dim3 grid((model->config.max_nodes + 255) / 256, model->config.num_heads);
    dim3 block(256);
    compute_attention<<<grid, block>>>(layer_input, model->attention_weights[layer_index], 
                                       model->attention_scores[layer_index], model->config.max_nodes, 
                                       model->config.hidden_features, model->config.num_heads);
    CUDA_CHECK(cudaGetLastError());

    // Apply attention (custom CUDA kernel)
    apply_attention<<<grid, block>>>(layer_input, model->attention_scores[layer_index], 
                                     layer_output, model->config.max_nodes, 
                                     model->config.hidden_features, model->config.num_heads);
    CUDA_CHECK(cudaGetLastError());

    // Linear transformation
    CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                model->config.hidden_features, model->config.max_nodes, model->config.hidden_features,
                &alpha, model->layer_weights[layer_index], model->config.hidden_features,
                layer_output, model->config.hidden_features,
                &beta, layer_output, model->config.hidden_features));

    // Add bias and apply activation (custom CUDA kernel)
    add_bias_activate<<<grid, block>>>(layer_output, model->layer_biases[layer_index], 
                                       model->config.max_nodes, model->config.hidden_features);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cublasDestroy(cublas));
}

// Backward pass for a single GAT layer
static void backward_gat_layer(GATModel* model, int layer_index, float* layer_input, float* layer_output, float* grad_output, int batch_size) {
    cublasHandle_t cublas;
    CUDA_CHECK(cublasCreate(&cublas));

    float alpha = 1.0f, beta = 0.0f;

    // Backpropagate through attention mechanism
    backward_attention<<<(model->config.max_nodes + 255) / 256, 256>>>(
        layer_input, model->attention_scores[layer_index], grad_output,
        model->d_attention_weights[layer_index], model->d_layer_outputs[layer_index],
        model->config.max_nodes, model->config.hidden_features, model->config.num_heads);
    CUDA_CHECK(cudaGetLastError());

    // Backpropagate through linear transformation
    CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                model->config.hidden_features, model->config.hidden_features, model->config.max_nodes,
                &alpha, model->d_layer_outputs[layer_index], model->config.hidden_features,
                model->layer_weights[layer_index], model->config.hidden_features,
                &beta, grad_output, model->config.hidden_features));

    CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                model->config.hidden_features, model->config.max_nodes, model->config.hidden_features,
                &alpha, model->layer_weights[layer_index], model->config.hidden_features,
                model->d_layer_outputs[layer_index], model->config.hidden_features,
                &beta, model->d_layer_weights[layer_index], model->config.hidden_features));

    // Compute gradient for biases
    CUDA_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                model->config.hidden_features, model->config.max_nodes,
                &alpha, model->d_layer_outputs[layer_index], model->config.hidden_features,
                model->ones, 1,
                &beta, model->d_layer_biases[layer_index], 1));

    // Backpropagate activation function
    backward_activation<<<(model->config.max_nodes * model->config.hidden_features + 255) / 256, 256>>>(
        layer_output, grad_output,
        model->config.max_nodes * model->config.hidden_features);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cublasDestroy(cublas));
}

// Compute final output (policy and value)
static void compute_output(GATModel* model, float* last_layer_output, float* out_pi, float* out_v, int batch_size) {
    cublasHandle_t cublas;
    CUDA_CHECK(cublasCreate(&cublas));

    float alpha = 1.0f, beta = 0.0f;

    // Output layer
    CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                model->config.num_actions, batch_size, model->config.hidden_features,
                &alpha, model->policy_weights, model->config.num_actions,
                last_layer_output, model->config.hidden_features,
                &beta, out_pi, model->config.num_actions));

    CUDA_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                1, model->config.hidden_features,
                &alpha, model->value_weights, 1,
                last_layer_output, 1,
                &beta, out_v, 1));

    // Apply softmax to policy output (custom CUDA kernel)
    dim3 grid_policy((model->config.num_actions + 255) / 256, batch_size);
    dim3 block_policy(256);
    softmax<<<grid_policy, block_policy>>>(out_pi, model->config.num_actions);
    CUDA_CHECK(cudaGetLastError());

    // Apply tanh to value output (custom CUDA kernel)
    dim3 grid_value((batch_size + 255) / 256);
    dim3 block_value(256);
    tanh_activate<<<grid_value, block_value>>>(out_v, batch_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cublasDestroy(cublas));
}

// Custom CUDA kernel implementations

__global__ void compute_attention(float* inputs, float* weights, float* scores, int num_nodes, int hidden_size, int num_heads) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;

    if (node_idx < num_nodes) {
        for (int i = 0; i < num_nodes; i++) {
            float score = 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                int input_idx = node_idx * hidden_size + j;
                int weight_idx = head_idx * 2 * hidden_size + j;
                score += inputs[input_idx] * weights[weight_idx];
                score += inputs[i * hidden_size + j] * weights[weight_idx + hidden_size];
            }
            scores[head_idx * num_nodes * num_nodes + node_idx * num_nodes + i] = score;
        }
    }
}

__global__ void apply_attention(float* inputs, float* scores, float* outputs, int num_nodes, int hidden_size, int num_heads) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int feature_idx = threadIdx.y;

    if (node_idx < num_nodes && feature_idx < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < num_nodes; i++) {
            int score_idx = head_idx * num_nodes * num_nodes + node_idx * num_nodes + i;
            int input_idx = i * hidden_size + feature_idx;
            sum += scores[score_idx] * inputs[input_idx];
        }
        outputs[head_idx * num_nodes * hidden_size + node_idx * hidden_size + feature_idx] = sum;
    }
}

__global__ void add_bias_activate(float* outputs, float* biases, int num_nodes, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes * hidden_size) {
        int feature_idx = idx % hidden_size;
        float x = outputs[idx] + biases[feature_idx];
        outputs[idx] = x > 0.0f ? x : 0.2f * x;  // LeakyReLU activation with alpha = 0.2
    }
}

__global__ void softmax(float* inputs, int size) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = (gid < size) ? inputs[gid] : -INFINITY;
    
    // Reduce to find max value
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(local_max, shared_data[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_data[0];
    __syncthreads();

    float local_sum = 0.0f;
    if (gid < size) {
        inputs[gid] = exp(inputs[gid] - max_val);
        local_sum = inputs[gid];
    }

    // Reduce to find sum
    shared_data[tid] = local_sum;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared_data[0];

    if (gid < size) {
        inputs[gid] /= sum;
    }
}

__global__ void tanh_activate(float* inputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        inputs[idx] = tanh(inputs[idx]);
    }
}

__global__ void softmax_cross_entropy_gradient(float* output, float* target, float* gradient, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        gradient[idx] = output[idx] - target[idx];
    }
}

__global__ void mse_gradient(float* output, float* target, float* gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradient[idx] = 2 * (output[idx] - target[idx]);
    }
}

__global__ void backward_attention(float* inputs, float* scores, float* grad_output, float* grad_weights, float* grad_inputs, int num_nodes, int hidden_size, int num_heads) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;

    if (node_idx < num_nodes) {
        for (int i = 0; i < num_nodes; i++) {
            float grad_score = 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                int grad_idx = head_idx * num_nodes * hidden_size + i * hidden_size + j;
                int input_idx = node_idx * hidden_size + j;
                grad_score += grad_output[grad_idx] * inputs[input_idx];
                atomicAdd(&grad_inputs[input_idx], grad_output[grad_idx] * scores[head_idx * num_nodes * num_nodes + i * num_nodes + node_idx]);
            }
            atomicAdd(&grad_weights[head_idx * 2 * hidden_size + node_idx], grad_score);
            atomicAdd(&grad_weights[head_idx * 2 * hidden_size + hidden_size + i], grad_score);
        }
    }
}

__global__ void backward_activation(float* inputs, float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] *= (inputs[idx] > 0.0f) ? 1.0f : 0.2f;  // LeakyReLU gradient
    }
}

// Compute initial gradient output for the backward pass
static void compute_initial_grad_output(GATModel* model, float* d_policy, float* d_value, float* grad_output, int batch_size) {
    cublasHandle_t cublas;
    CUDA_CHECK(cublasCreate(&cublas));

    float alpha = 1.0f, beta = 0.0f;

    // Compute gradient for policy head
    CUDA_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                model->config.hidden_features, batch_size, model->config.num_actions,
                &alpha, model->policy_weights, model->config.hidden_features,
                d_policy, model->config.num_actions,
                &beta, grad_output, model->config.hidden_features));

    // Compute gradient for value head and add it to grad_output
    CUDA_CHECK(cublasSger(cublas, model->config.hidden_features, batch_size,
               &alpha, model->value_weights, 1,
               d_value, 1,
               grad_output, model->config.hidden_features));

    CUDA_CHECK(cublasDestroy(cublas));

    // Apply activation gradient (LeakyReLU)
    dim3 grid((batch_size * model->config.hidden_features + 255) / 256);
    dim3 block(256);
    backward_activation<<<grid, block>>>(model->layer_outputs[model->config.num_layers - 1],
                                         grad_output,
                                         batch_size * model->config.hidden_features);
    CUDA_CHECK(cudaGetLastError());
}