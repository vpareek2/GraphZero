/**
 * @file players.cuh
 * @brief Player Interfaces and Implementations Header File
 */

#ifndef PLAYERS_CUH
#define PLAYERS_CUH

#include "../games/game.cuh"
#include "../mcts/mcts.cuh"
#include "../networks/gat/gat.cuh"

/**
 * @struct IPlayer
 * @brief Interface for a player
 */
typedef struct {
    int (*get_action)(const IGame* game, const int* board, int player);
} IPlayer;

/**
 * @struct RandomPlayer
 * @brief Implements a random player
 */
typedef struct {
    IPlayer base;
} RandomPlayer;

/**
 * @struct MCTSPlayer
 * @brief Implements a Monte Carlo Tree Search player
 */
typedef struct {
    IPlayer base;
    MCTSState* mcts_state;
    float temperature;
} MCTSPlayer;

/**
 * @struct NNetPlayer
 * @brief Implements a Neural Network player
 */
typedef struct {
    IPlayer base;
    INeuralNet* net;
    MCTSState* mcts_state;
    float temperature;
} NNetPlayer;

/**
 * @brief Creates a random player
 * @return Pointer to the created RandomPlayer
 */
RandomPlayer* create_random_player();

/**
 * @brief Creates an MCTS player
 * @param game Pointer to the game
 * @param num_simulations Number of simulations for MCTS
 * @param temperature Temperature for action selection
 * @return Pointer to the created MCTSPlayer
 */
MCTSPlayer* create_mcts_player(
    IGame* game,
    int num_simulations,
    float temperature
);

/**
 * @brief Creates a Neural Network player
 * @param game Pointer to the game
 * @param net Pointer to the neural network
 * @param num_simulations Number of simulations for MCTS
 * @param temperature Temperature for action selection
 * @return Pointer to the created NNetPlayer
 */
NNetPlayer* create_nnet_player(
    IGame* game,
    INeuralNet* net,
    int num_simulations,
    float temperature
);

/**
 * @brief Destroys a random player
 * @param player Pointer to the RandomPlayer to destroy
 */
void destroy_random_player(
    RandomPlayer* player
);

/**
 * @brief Destroys an MCTS player
 * @param player Pointer to the MCTSPlayer to destroy
 */
void destroy_mcts_player(
    MCTSPlayer* player
);

/**
 * @brief Destroys a Neural Network player
 * @param player Pointer to the NNetPlayer to destroy
 */
void destroy_nnet_player(
    NNetPlayer* player
);

#endif // PLAYERS_CUH