#ifndef CONNECT4_CUH
#define CONNECT4_CUH

#include "../game.h"

#define CONNECT4_ROWS 6
#define CONNECT4_COLS 7
#define CONNECT4_BOARD_SIZE (CONNECT4_ROWS * CONNECT4_COLS)
#define CONNECT4_ACTION_SIZE CONNECT4_COLS

typedef struct {
    IGame base;
} Connect4Game;

__host__ __device__ void connect4_init(IGame* self);
__host__ __device__ void connect4_get_init_board(const IGame* self, int* board);
__host__ __device__ void connect4_get_board_size(const IGame* self, int* rows, int* cols);
__host__ __device__ int connect4_get_action_size(const IGame* self);
__host__ __device__ void connect4_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);
__host__ __device__ void connect4_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves);
__host__ __device__ int connect4_get_game_ended(const IGame* self, const int* board, int player);
__host__ __device__ void connect4_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board);
__host__ __device__ float connect4_evaluate(const IGame* self, const int* board, int player);

// CPU-only functions
void connect4_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries);
void connect4_string_representation(const IGame* self, const int* board, char* str, int str_size);
void connect4_display(const IGame* self, const int* board);

// Create a Connect4 game instance
Connect4Game* create_connect4_game();

#endif // CONNECT4_CUH