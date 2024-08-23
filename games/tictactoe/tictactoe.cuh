#ifndef TICTACTOE_CUH
#define TICTACTOE_CUH

#include "../game.h"

#define TICTACTOE_BOARD_SIZE 3
#define TICTACTOE_NUM_SQUARES (TICTACTOE_BOARD_SIZE * TICTACTOE_BOARD_SIZE)

typedef struct {
    IGame base;
} TicTacToeGame;

__host__ __device__ void tictactoe_init(IGame* self);
__host__ __device__ void tictactoe_get_init_board(const IGame* self, int* board);
__host__ __device__ void tictactoe_get_board_size(const IGame* self, int* rows, int* cols);
__host__ __device__ int tictactoe_get_action_size(const IGame* self);
__host__ __device__ void tictactoe_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);
__host__ __device__ void tictactoe_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves);
__host__ __device__ int tictactoe_get_game_ended(const IGame* self, const int* board, int player);
__host__ __device__ void tictactoe_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board);
__host__ __device__ float tictactoe_evaluate(const IGame* self, const int* board, int player);

// CPU-only functions
void tictactoe_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries);
void tictactoe_string_representation(const IGame* self, const int* board, char* str, int str_size);
void tictactoe_display(const IGame* self, const int* board);

// Create a TicTacToe game instance
TicTacToeGame* create_tictactoe_game();

#endif // TICTACTOE_CUH