#include "connect4.cuh"
#include <stdio.h>
#include <string.h>

__host__ __device__ void connect4_init(IGame* self) {
    // No initialization needed for Connect4
}

__host__ __device__ void connect4_get_init_board(const IGame* self, int* board) {
    memset(board, 0, CONNECT4_BOARD_SIZE * sizeof(int));
}

__host__ __device__ void connect4_get_board_size(const IGame* self, int* rows, int* cols) {
    *rows = CONNECT4_ROWS;
    *cols = CONNECT4_COLS;
}

__host__ __device__ int connect4_get_action_size(const IGame* self) {
    return CONNECT4_ACTION_SIZE;
}

__host__ __device__ void connect4_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player) {
    memcpy(next_board, board, CONNECT4_BOARD_SIZE * sizeof(int));
    
    // Find the lowest empty row in the selected column
    for (int row = CONNECT4_ROWS - 1; row >= 0; row--) {
        if (next_board[row * CONNECT4_COLS + action] == 0) {
            next_board[row * CONNECT4_COLS + action] = player;
            break;
        }
    }
    
    *next_player = -player;
}

__host__ __device__ void connect4_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves) {
    for (int col = 0; col < CONNECT4_COLS; col++) {
        valid_moves[col] = (board[col] == 0);
    }
}

__host__ __device__ int connect4_get_game_ended(const IGame* self, const int* board, int player) {
    // Check for a win
    for (int row = 0; row < CONNECT4_ROWS; row++) {
        for (int col = 0; col < CONNECT4_COLS; col++) {
            if (board[row * CONNECT4_COLS + col] == player) {
                // Check horizontal
                if (col <= CONNECT4_COLS - 4 &&
                    board[row * CONNECT4_COLS + col + 1] == player &&
                    board[row * CONNECT4_COLS + col + 2] == player &&
                    board[row * CONNECT4_COLS + col + 3] == player) {
                    return 1;
                }
                // Check vertical
                if (row <= CONNECT4_ROWS - 4 &&
                    board[(row + 1) * CONNECT4_COLS + col] == player &&
                    board[(row + 2) * CONNECT4_COLS + col] == player &&
                    board[(row + 3) * CONNECT4_COLS + col] == player) {
                    return 1;
                }
                // Check diagonal (down-right)
                if (col <= CONNECT4_COLS - 4 && row <= CONNECT4_ROWS - 4 &&
                    board[(row + 1) * CONNECT4_COLS + col + 1] == player &&
                    board[(row + 2) * CONNECT4_COLS + col + 2] == player &&
                    board[(row + 3) * CONNECT4_COLS + col + 3] == player) {
                    return 1;
                }
                // Check diagonal (up-right)
                if (col <= CONNECT4_COLS - 4 && row >= 3 &&
                    board[(row - 1) * CONNECT4_COLS + col + 1] == player &&
                    board[(row - 2) * CONNECT4_COLS + col + 2] == player &&
                    board[(row - 3) * CONNECT4_COLS + col + 3] == player) {
                    return 1;
                }
            }
        }
    }

    // Check for a draw
    for (int col = 0; col < CONNECT4_COLS; col++) {
        if (board[col] == 0) {
            return 0; // Game is not over
        }
    }
    
    return 0.00001; // Draw
}

__host__ __device__ void connect4_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board) {
    for (int i = 0; i < CONNECT4_BOARD_SIZE; i++) {
        canonical_board[i] = board[i] * player;
    }
}

__host__ __device__ float connect4_evaluate(const IGame* self, const int* board, int player) {
    int game_result = connect4_get_game_ended(self, board, player);
    return (float)game_result * player;
}

// CPU-only functions
void connect4_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries) {
    // Connect4 only has left-right symmetry
    *num_symmetries = 2;
    
    // Original board and pi
    memcpy(symmetries[0], board, CONNECT4_BOARD_SIZE * sizeof(int));
    memcpy(symmetries_pi[0], pi, CONNECT4_ACTION_SIZE * sizeof(float));
    
    // Left-right symmetry
    for (int row = 0; row < CONNECT4_ROWS; row++) {
        for (int col = 0; col < CONNECT4_COLS; col++) {
            symmetries[1][row * CONNECT4_COLS + col] = board[row * CONNECT4_COLS + (CONNECT4_COLS - 1 - col)];
        }
    }
    for (int col = 0; col < CONNECT4_COLS; col++) {
        symmetries_pi[1][col] = pi[CONNECT4_COLS - 1 - col];
    }
}

void connect4_string_representation(const IGame* self, const int* board, char* str, int str_size) {
    char symbols[3] = {'.', 'X', 'O'};
    int pos = 0;
    
    for (int row = 0; row < CONNECT4_ROWS; row++) {
        for (int col = 0; col < CONNECT4_COLS; col++) {
            int piece = board[row * CONNECT4_COLS + col];
            str[pos++] = symbols[piece + 1];
            if (pos >= str_size - 1) goto end;
        }
        str[pos++] = '\n';
        if (pos >= str_size - 1) goto end;
    }
    
end:
    str[pos] = '\0';
}

void connect4_display(const IGame* self, const int* board) {
    char str[CONNECT4_BOARD_SIZE + CONNECT4_ROWS + 1];
    connect4_string_representation(self, board, str, sizeof(str));
    printf("%s\n", str);
}

Connect4Game* create_connect4_game() {
    Connect4Game* game = (Connect4Game*)malloc(sizeof(Connect4Game));
    game->base.init = connect4_init;
    game->base.get_init_board = connect4_get_init_board;
    game->base.get_board_size = connect4_get_board_size;
    game->base.get_action_size = connect4_get_action_size;
    game->base.get_next_state = connect4_get_next_state;
    game->base.get_valid_moves = connect4_get_valid_moves;
    game->base.get_game_ended = connect4_get_game_ended;
    game->base.get_canonical_form = connect4_get_canonical_form;
    game->base.get_symmetries = connect4_get_symmetries;
    game->base.string_representation = connect4_string_representation;
    game->base.display = connect4_display;
    game->base.evaluate = connect4_evaluate;

    // CUDA-compatible functions
    game->base.get_init_board_cuda = connect4_get_init_board;
    game->base.get_next_state_cuda = connect4_get_next_state;
    game->base.get_valid_moves_cuda = connect4_get_valid_moves;
    game->base.get_game_ended_cuda = connect4_get_game_ended;
    game->base.get_canonical_form_cuda = connect4_get_canonical_form;
    game->base.evaluate_cuda = connect4_evaluate;

    return game;
}