#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include "games/connect4/connect4.cuh"
#include "games/tictactoe/tictactoe.cuh"
#include "networks/gat/gat.cuh"
#include "self_play/self_play.cuh"
#include "utils/cuda_utils.cuh"
#include "config.h"

#define CUDA_CHECK(call) { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(1); \
    } \
}

#define NCCL_CHECK(call) { \
    ncclResult_t status = call; \
    if (status != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(status)); \
        exit(1); \
    } \
}

int main(int argc, char* argv[]) {
    int world_size, world_rank;
    ncclUniqueId nccl_id;
    ncclComm_t nccl_comm;
    cudaStream_t cuda_stream;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Ensure we have 8 processes
    if (world_size != 8) {
        fprintf(stderr, "This application must be run with 8 MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Load configuration
    GlobalConfig* config = load_config("config.json");
    if (!config) {
        fprintf(stderr, "Failed to load configuration\n");
        MPI_Finalize();
        return 1;
    }

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(world_rank));

    // Initialize NCCL
    if (world_rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank));

    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&cuda_stream));

    // Parse command-line arguments
    const char* game_type = "tictactoe";
    if (argc > 1) {
        game_type = argv[1];
    }

    // Create game instance
    IGame* game = NULL;
    if (strcmp(game_type, "tictactoe") == 0) {
        game = (IGame*)create_tictactoe_game();
    } else if (strcmp(game_type, "connect4") == 0) {
        game = (IGame*)create_connect4_game();
    } else {
        fprintf(stderr, "Unsupported game type: %s\n", game_type);
        free_config(config);
        MPI_Finalize();
        return 1;
    }

    if (!game) {
        fprintf(stderr, "Failed to create game instance\n");
        free_config(config);
        MPI_Finalize();
        return 1;
    }

    // Create neural network instance
    INeuralNet* nnet = create_gat_model(game);
    if (!nnet) {
        fprintf(stderr, "Failed to create GAT neural network instance\n");
        game->destroy(game);
        free_config(config);
        MPI_Finalize();
        return 1;
    }

    // Use the configuration for self-play
    SelfPlayConfig sp_config = config->self_play.config;

    // Create self-play pipeline
    SelfPlayPipeline* pipeline = create_self_play_pipeline(game, nnet, sp_config);
    if (!pipeline) {
        fprintf(stderr, "Failed to create self-play pipeline\n");
        nnet->destroy(nnet);
        game->destroy(game);
        free_config(config);
        MPI_Finalize();
        return 1;
    }

    // Main training loop
    for (int i = 1; i <= sp_config.numIters; i++) {
        if (world_rank == 0) {
            printf("Starting iteration %d\n", i);
        }

        // Execute self-play (distributed across GPUs)
        execute_self_play_distributed(pipeline, world_rank, world_size, nccl_comm, cuda_stream);

        // Synchronize and aggregate experiences
        MPI_Barrier(MPI_COMM_WORLD);

        // Train neural network (distributed across GPUs)
        learn_distributed(pipeline, world_rank, world_size, nccl_comm, cuda_stream);

        // Optionally save checkpoint (only on rank 0)
        if (i % 10 == 0 && world_rank == 0) {
            char filename[config->neural_network.max_filename_length];
            snprintf(filename, sizeof(filename), "checkpoint_%04d.pth", i);
            nnet->save_checkpoint(nnet, sp_config.checkpoint, filename);
            printf("Saved checkpoint: %s\n", filename);
        }

        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Clean up
    destroy_self_play_pipeline(pipeline);
    nnet->destroy(nnet);
    game->destroy(game);
    free_config(config);

    // Clean up NCCL and CUDA
    ncclCommDestroy(nccl_comm);
    cudaStreamDestroy(cuda_stream);

    // Reset CUDA device
    CUDA_CHECK(cudaDeviceReset());

    // Finalize MPI
    MPI_Finalize();

    if (world_rank == 0) {
        printf("Training completed successfully\n");
    }
    return 0;
}