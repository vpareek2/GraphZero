/**
 * @file mcts.cuh
 * @brief Monte Carlo Tree Search Header File
 */

#ifndef MCTS_CUH
#define MCTS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../games/game.cuh"

// Define constants
#define MAX_BOARD_SIZE     64
#define MAX_CHILDREN       218
#define C_PUCT             1.0
#define NUM_SIMULATIONS    1600
#define THREADS_PER_BLOCK  256
#define MAX_BATCH_SIZE     64  // Constant for batch processing

/**
 * @struct MCTSNode
 * @brief Represents a node in the Monte Carlo Tree Search
 */
typedef struct MCTSNode {
    int             board[MAX_BOARD_SIZE];
    MCTSNode*       children[MAX_CHILDREN];
    int             num_children;
    float           P[MAX_CHILDREN];
    float           Q[MAX_CHILDREN];
    int             N[MAX_CHILDREN];
    int             visit_count;
    float           value_sum;
    int             player;
    int             action;
} MCTSNode;

/**
 * @struct MCTSState
 * @brief Represents the state of the Monte Carlo Tree Search
 */
typedef struct MCTSState {
    IGame*          game;
    MCTSNode*       root;
} MCTSState;

// CPU functions

/**
 * @brief Initializes the MCTS state
 * @param game Game instance
 * @return New MCTS state
 */
MCTSState*        mcts_init(
    IGame*          game
);

/**
 * @brief Frees the MCTS state
 * @param state MCTS state to free
 */
void              mcts_free(
    MCTSState*      state
);

/**
 * @brief Selects an action based on the MCTS
 * @param state MCTS state
 * @param temperature Temperature for action selection
 * @return Selected action
 */
int               mcts_select_action(
    MCTSState*      state,
    float           temperature
);

/**
 * @brief Updates the MCTS state with a move
 * @param state MCTS state
 * @param action Action to update with
 */
void              mcts_update_with_move(
    MCTSState*      state,
    int             action
);

// CUDA kernel functions

/**
 * @brief CUDA kernel for MCTS simulation
 */
__global__ void   mcts_simulate_kernel(
    MCTSNode*       nodes,
    int*            boards,
    int*            players,
    curandState*    rng_states,
    IGame*          game,
    int             num_games
);

/**
 * @brief Device function for MCTS simulation
 */
__device__ float  mcts_simulate(
    MCTSNode*       node,
    int*            board,
    int             player,
    curandState*    rng_state,
    IGame*          game
);

/**
 * @brief Device function for node selection in MCTS
 */
__device__ MCTSNode* mcts_select(
    MCTSNode*       node
);

/**
 * @brief Device function for node expansion in MCTS
 */
__device__ void   mcts_expand(
    MCTSNode*       node,
    int*            board,
    int             player,
    IGame*          game
);

/**
 * @brief Device function for board evaluation in MCTS
 */
__device__ float  mcts_evaluate(
    int*            board,
    int             player,
    IGame*          game
);

/**
 * @brief Device function for backpropagation in MCTS
 */
__device__ void   mcts_backpropagate(
    MCTSNode*       node,
    float           value
);

// CUDA helper functions

/**
 * @brief CUDA kernel for initializing random number generators
 */
__global__ void   init_rng(
    curandState*    states,
    unsigned long   seed,
    int             num_states
);

/**
 * @brief Runs MCTS for a batch of games
 * @param states Array of MCTS states
 * @param num_games Number of games in the batch
 */
void              mcts_run_batch(
    MCTSState**     states,
    int             num_games
);

#endif // MCTS_CUH