#include "tictactoe.cuh"
#include <stdio.h>
#include <string.h>

__host__ __device__ void tictactoe_init(IGame* self) {
    // No initialization needed for TicTacToe
}

__host__ __device__ void tictactoe_get_init_board(const IGame* self, int* board) {
    for (int i = 0; i < TICTACTOE_NUM_SQUARES; i++) {
        board[i] = 0;
    }
}

__host__ __device__ void tictactoe_get_board_size(const IGame* self, int* rows, int* cols) {
    *rows = TICTACTOE_BOARD_SIZE;
    *cols = TICTACTOE_BOARD_SIZE;
}

__host__ __device__ int tictactoe_get_action_size(const IGame* self) {
    return TICTACTOE_NUM_SQUARES;
}

__host__ __device__ void tictactoe_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player) {
    for (int i = 0; i < TICTACTOE_NUM_SQUARES; i++) {
        next_board[i] = board[i];
    }
    next_board[action] = player;
    *next_player = -player;
}

__host__ __device__ void tictactoe_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves) {
    for (int i = 0; i < TICTACTOE_NUM_SQUARES; i++) {
        valid_moves[i] = (board[i] == 0);
    }
}

__host__ __device__ int tictactoe_get_game_ended(const IGame* self, const int* board, int player) {
    // Check rows, columns, and diagonals
    for (int i = 0; i < TICTACTOE_BOARD_SIZE; i++) {
        if (board[i*TICTACTOE_BOARD_SIZE] != 0 &&
            board[i*TICTACTOE_BOARD_SIZE] == board[i*TICTACTOE_BOARD_SIZE + 1] &&
            board[i*TICTACTOE_BOARD_SIZE] == board[i*TICTACTOE_BOARD_SIZE + 2]) {
            return board[i*TICTACTOE_BOARD_SIZE] * player;
        }
        if (board[i] != 0 &&
            board[i] == board[i + TICTACTOE_BOARD_SIZE] &&
            board[i] == board[i + 2*TICTACTOE_BOARD_SIZE]) {
            return board[i] * player;
        }
    }
    if (board[0] != 0 && board[0] == board[4] && board[0] == board[8]) {
        return board[0] * player;
    }
    if (board[2] != 0 && board[2] == board[4] && board[2] == board[6]) {
        return board[2] * player;
    }

    // Check for draw
    for (int i = 0; i < TICTACTOE_NUM_SQUARES; i++) {
        if (board[i] == 0) {
            return 0; // Game is not over
        }
    }
    return 1e-4; // Draw
}

__host__ __device__ void tictactoe_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board) {
    for (int i = 0; i < TICTACTOE_NUM_SQUARES; i++) {
        canonical_board[i] = board[i] * player;
    }
}

__host__ __device__ float tictactoe_evaluate(const IGame* self, const int* board, int player) {
    int game_result = tictactoe_get_game_ended(self, board, player);
    if (game_result == 1) return 1.0f;
    if (game_result == -1) return -1.0f;
    if (game_result == 1e-4) return 0.0f;

    // Simple heuristic evaluation if the game hasn't ended
    float score = 0.0f;
    int lines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  // Rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  // Columns
        {0, 4, 8}, {2, 4, 6}  // Diagonals
    };
    
    for (int i = 0; i < 8; i++) {
        int player_count = 0;
        int opponent_count = 0;
        for (int j = 0; j < 3; j++) {
            if (board[lines[i][j]] == player) player_count++;
            else if (board[lines[i][j]] == -player) opponent_count++;
        }
        
        if (player_count > 0 && opponent_count == 0) {
            score += 0.1f * player_count;
        } else if (opponent_count > 0 && player_count == 0) {
            score -= 0.1f * opponent_count;
        }
    }
    
    return score;
}

// CPU-only functions
void tictactoe_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries) {
    *num_symmetries = 8;
    
    // Original
    memcpy(symmetries[0], board, TICTACTOE_NUM_SQUARES * sizeof(int));
    memcpy(symmetries_pi[0], pi, TICTACTOE_NUM_SQUARES * sizeof(float));
    
    // Rotate 90, 180, 270
    for (int r = 1; r < 4; r++) {
        for (int i = 0; i < TICTACTOE_BOARD_SIZE; i++) {
            for (int j = 0; j < TICTACTOE_BOARD_SIZE; j++) {
                int new_i, new_j;
                switch (r) {
                    case 1: new_i = j; new_j = TICTACTOE_BOARD_SIZE-1-i; break;
                    case 2: new_i = TICTACTOE_BOARD_SIZE-1-i; new_j = TICTACTOE_BOARD_SIZE-1-j; break;
                    case 3: new_i = TICTACTOE_BOARD_SIZE-1-j; new_j = i; break;
                }
                symmetries[r][TICTACTOE_BOARD_SIZE*new_i + new_j] = board[TICTACTOE_BOARD_SIZE*i + j];
                symmetries_pi[r][TICTACTOE_BOARD_SIZE*new_i + new_j] = pi[TICTACTOE_BOARD_SIZE*i + j];
            }
        }
    }
    
    // Flip horizontal, vertical, diagonal, anti-diagonal
    for (int f = 0; f < 4; f++) {
        for (int i = 0; i < TICTACTOE_BOARD_SIZE; i++) {
            for (int j = 0; j < TICTACTOE_BOARD_SIZE; j++) {
                int new_i, new_j;
                switch (f) {
                    case 0: new_i = i; new_j = TICTACTOE_BOARD_SIZE-1-j; break;
                    case 1: new_i = TICTACTOE_BOARD_SIZE-1-i; new_j = j; break;
                    case 2: new_i = j; new_j = i; break;
                    case 3: new_i = TICTACTOE_BOARD_SIZE-1-j; new_j = TICTACTOE_BOARD_SIZE-1-i; break;
                }
                symmetries[f+4][TICTACTOE_BOARD_SIZE*new_i + new_j] = board[TICTACTOE_BOARD_SIZE*i + j];
                symmetries_pi[f+4][TICTACTOE_BOARD_SIZE*new_i + new_j] = pi[TICTACTOE_BOARD_SIZE*i + j];
            }
        }
    }
}

void tictactoe_string_representation(const IGame* self, const int* board, char* str, int str_size) {
    char symbols[3] = {'.', 'X', 'O'};
    int pos = 0;
    
    for (int i = 0; i < TICTACTOE_BOARD_SIZE; i++) {
        for (int j = 0; j < TICTACTOE_BOARD_SIZE; j++) {
            str[pos++] = symbols[board[i*TICTACTOE_BOARD_SIZE + j] + 1];
            if (pos >= str_size - 1) goto end;
        }
        str[pos++] = '\n';
        if (pos >= str_size - 1) goto end;
    }
    
end:
    str[pos] = '\0';
}

void tictactoe_display(const IGame* self, const int* board) {
    char str[TICTACTOE_NUM_SQUARES + TICTACTOE_BOARD_SIZE + 1];
    tictactoe_string_representation(self, board, str, sizeof(str));
    printf("%s\n", str);
}

TicTacToeGame* create_tictactoe_game() {
    TicTacToeGame* game = (TicTacToeGame*)malloc(sizeof(TicTacToeGame));
    game->base.init = tictactoe_init;
    game->base.get_init_board = tictactoe_get_init_board;
    game->base.get_board_size = tictactoe_get_board_size;
    game->base.get_action_size = tictactoe_get_action_size;
    game->base.get_next_state = tictactoe_get_next_state;
    game->base.get_valid_moves = tictactoe_get_valid_moves;
    game->base.get_game_ended = tictactoe_get_game_ended;
    game->base.get_canonical_form = tictactoe_get_canonical_form;
    game->base.get_symmetries = tictactoe_get_symmetries;
    game->base.string_representation = tictactoe_string_representation;
    game->base.display = tictactoe_display;
    game->base.evaluate = tictactoe_evaluate;

    // CUDA-compatible functions
    game->base.get_init_board_cuda = tictactoe_get_init_board;
    game->base.get_next_state_cuda = tictactoe_get_next_state;
    game->base.get_valid_moves_cuda = tictactoe_get_valid_moves;
    game->base.get_game_ended_cuda = tictactoe_get_game_ended;
    game->base.get_canonical_form_cuda = tictactoe_get_canonical_form;
    game->base.evaluate_cuda = tictactoe_evaluate;

    return game;
}