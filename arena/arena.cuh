/**
 * @file arena.cuh
 * @brief Arena Header File
 */

#ifndef ARENA_CUH
#define ARENA_CUH

// Include player and game headers
#include "../player/player.cuh"
#include "../game/game.cuh"

// Define maximum board size
#define MAX_BOARD_SIZE         64  // Assuming the maximum board size is for 8x8 chess

/**
 * @struct Arena
 * @brief Represents an arena for playing games
 */
typedef struct Arena {
    Player*          player1;      // Player 1
    Player*          player2;      // Player 2
    IGame*           game;         // Game instance
    int*             board;
    int*             board_size;
} Arena;

/**
 * @brief Creates a new arena
 * @param player1 Player 1
 * @param player2 Player 2
 * @param game Game instance
 * @return New arena instance
 */
Arena*            create_arena(
    Player*          player1,
    Player*          player2,
    IGame*           game
);

/**
 * @brief Plays a single game in the arena
 * @param arena Arena instance
 * @param verbose Enable verbose output
 * @return Game result
 */
int              play_game(
    Arena*          arena,
    bool            verbose
);

/**
 * @brief Plays multiple games in the arena
 * @param arena Arena instance
 * @param num_games Number of games to play
 * @param wins Wins counter
 * @param losses Losses counter
 * @param draws Draws counter
 */
void             play_games(
    Arena*          arena,
    int             num_games,
    int*            wins,
    int*            losses,
    int*            draws
);

/**
 * @brief Destroys an arena instance
 * @param arena Arena instance
 */
void             destroy_arena(
    Arena*          arena
);

#endif // ARENA_CUH