/**
 * @file self_play.cuh
 * @brief Self-Play Pipeline Header File
 */

#ifndef SELF_PLAY_CUH
#define SELF_PLAY_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mpi.h>
#include <nccl.h>

#include "../games/game.cuh"
#include "../networks/neural_network.h"
#include "../mcts/mcts.cuh"

// Define constants
#define MAX_BATCH_SIZE             1024
#define MAX_NUM_GAMES              10000
#define MAX_GAME_LENGTH            1000
#define TERMINAL_STATE             -1
#define MAX_FILENAME_LENGTH        256

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(1); \
    } \
}

/**
 * @struct SelfPlayConfig
 * @brief Configuration for self-play
 */
typedef struct SelfPlayConfig {
    int             numIters;
    int             numEps;
    int             numGames;
    int             batchSize;
    int             numMCTSSims;
    float           tempThreshold;
    float           updateThreshold;
    int             maxlenOfQueue;
    int             numItersForTrainExamplesHistory;
    int             arenaCompare;
    char            checkpoint[MAX_FILENAME_LENGTH];
} SelfPlayConfig;

/**
 * @struct TrainingExample
 * @brief Represents a single training example
 */
typedef struct TrainingExample {
    int             board[MAX_BOARD_SIZE];
    float           pi[MAX_BOARD_SIZE];
    float           v;
} TrainingExample;

/**
 * @struct SelfPlayPipeline
 * @brief Main structure for the self-play pipeline
 */
typedef struct SelfPlayPipeline {
    IGame*          game;
    INeuralNet*     nnet;
    INeuralNet*     pnet;
    MCTSState*      mcts;
    SelfPlayConfig  config;
    TrainingExample** trainExamplesHistory;
    int             historySize;
    int             skipFirstSelfPlay;

    // GPU resources
    curandState*    d_rng_states;
    int*            d_boards;
    float*          d_pis;
    float*          d_vs;
    MCTSNode*       d_mcts_nodes;
    TrainingExample* d_examples;
} SelfPlayPipeline;

// Function prototypes

/**
 * @brief Creates a new self-play pipeline
 * @param game Game instance
 * @param nnet Neural network
 * @param config Self-play configuration
 * @return New self-play pipeline instance
 */
SelfPlayPipeline* create_self_play_pipeline(
    IGame*          game,
    INeuralNet*     nnet,
    SelfPlayConfig  config
);

/**
 * @brief Destroys a self-play pipeline
 * @param pipeline Self-play pipeline to destroy
 */
void              destroy_self_play_pipeline(
    SelfPlayPipeline* pipeline
);

/**
 * @brief Executes the self-play process
 * @param pipeline Self-play pipeline
 */
void              execute_self_play(
    SelfPlayPipeline* pipeline
);

/**
 * @brief Executes the learning process
 * @param pipeline Self-play pipeline
 */
void              learn(
    SelfPlayPipeline* pipeline
);

void execute_self_play_distributed(
    SelfPlayPipeline* pipeline,
    int world_rank,
    int world_size,
    ncclComm_t nccl_comm,
    cudaStream_t cuda_stream
);

void learn_distributed(
    SelfPlayPipeline* pipeline,
    int world_rank,
    int world_size,
    ncclComm_t nccl_comm,
    cudaStream_t cuda_stream
);

// CUDA kernel function prototypes

/**
 * @brief Initializes random number generator states
 * @param states RNG states
 * @param seed Random seed
 * @param num_states Number of states to initialize
 */
__global__ void   init_rng(
    curandState*    states,
    unsigned long   seed,
    int             num_states
);

/**
 * @brief Parallel self-play kernel
 * @param roots MCTS root nodes
 * @param boards Game boards
 * @param pis Policy vectors
 * @param vs Value predictions
 * @param players Current players
 * @param rng_states RNG states
 * @param game Game instance
 * @param nnet Neural network
 * @param num_games Number of games
 * @param num_mcts_sims Number of MCTS simulations
 * @param temp_threshold Temperature threshold
 * @param examples Training examples
 */
__global__ void   parallel_self_play_kernel(
    MCTSNode*       roots,
    int*            boards,
    float*          pis,
    float*          vs,
    int*            players,
    curandState*    rng_states,
    IGame*          game,
    INeuralNet*     nnet,
    int             num_games,
    int             num_mcts_sims,
    int             temp_threshold,
    TrainingExample* examples
);

// Helper function prototypes

void              add_to_training_history(SelfPlayPipeline* pipeline, TrainingExample* examples, int num_examples);
void              save_train_examples(SelfPlayPipeline* pipeline, int iteration);
void              load_train_examples(SelfPlayPipeline* pipeline);
int               pit_against_previous_version(SelfPlayPipeline* pipeline);
void              get_checkpoint_file(SelfPlayPipeline* pipeline, int iteration, char* filename);

// MCTS and action selection helpers
__device__ void   mcts_simulate(MCTSNode* node, int* board, int player, curandState* rng_state, IGame* game, INeuralNet* nnet);
__device__ void   mcts_get_policy(MCTSNode* node, float* policy, float temperature);
__device__ int    select_action(float* policy, int action_size, curandState* rng_state);
__device__ MCTSNode* mcts_move_to_child(MCTSNode* node, int action);
__device__ void   mcts_expand(MCTSNode* node, int* board, int player, IGame* game);
__device__ MCTSNode* mcts_select_uct(MCTSNode* node);

#endif // SELF_PLAY_CUH