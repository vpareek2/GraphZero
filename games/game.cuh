#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include <cuda_runtime.h>

#define MAX_BOARD_SIZE 64  // Suitable for games up to 8x8 (planning chess scale up)
#define MAX_SYMMETRIES 8   // Change later

typedef struct IGame IGame;

struct IGame {
    // CPU and CUDA-compatible functions
    __host__ __device__ void (*init)(IGame* self);
    __host__ __device__ void (*get_init_board)(const IGame* self, int* board);
    __host__ __device__ void (*get_board_size)(const IGame* self, int* rows, int* cols);
    __host__ __device__ int (*get_action_size)(const IGame* self);
    __host__ __device__ void (*get_next_state)(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);
    __host__ __device__ void (*get_valid_moves)(const IGame* self, const int* board, int player, bool* valid_moves);
    __host__ __device__ int (*get_game_ended)(const IGame* self, const int* board, int player);
    __host__ __device__ void (*get_canonical_form)(const IGame* self, const int* board, int player, int* canonical_board);
    __host__ __device__ float (*evaluate)(const IGame* self, const int* board, int player);

    // CPU-only functions
    void (*get_symmetries)(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries);
    void (*string_representation)(const IGame* self, const int* board, char* str, int str_size);
    void (*display)(const IGame* self, const int* board);
};

#endif // GAME_H