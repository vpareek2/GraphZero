#ifndef CHESS_H
#define CHESS_H

#include "../game.cuh"
#include <cuda_runtime.h>

#define CHESS_BOARD_SIZE 64
#define NUM_PIECE_TYPES 6
#define NUM_PLAYERS 2

// Piece representation
#define EMPTY 0
#define PAWN 1
#define KNIGHT 2
#define BISHOP 3
#define ROOK 4
#define QUEEN 5
#define KING 6

// Colors
#define WHITE 1
#define BLACK -1

// For bit ops
#define LSB(x) (__builtin_ffsll(x) - 1)
#define POP_LSB(x) (x &= (x - 1))

// Zobrist hashing constants
#define ZOBRIST_PIECE_TYPES 12  // 6 piece types * 2 colors
#define ZOBRIST_SQUARES 64

typedef struct {
    int pieces[CHESS_BOARD_SIZE];
    int player;
    bool castling_rights[2][2]; // [player][kingside/queenside]
    int en_passant_target;
    int halfmove_clock;
    int fullmove_number;
    unsigned long long position_history[MAX_HISTORY];
    int history_count;
} ChessBoard;

typedef struct {
    IGame base;
    ChessBoard board;
} ChessGame;

// Function prototypes
__host__ __device__ void chess_init(IGame* self);
__host__ __device__ void chess_get_init_board(const IGame* self, int* board);
__host__ __device__ void chess_get_board_size(const IGame* self, int* rows, int* cols);
__host__ __device__ int chess_get_action_size(const IGame* self);
__host__ __device__ void chess_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player);
__host__ __device__ void chess_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves);
__host__ __device__ int chess_get_game_ended(const IGame* self, const int* board, int player);
__host__ __device__ void chess_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board);
__host__ __device__ float chess_evaluate(const IGame* self, const int* board, int player);

// CPU-only functions
void chess_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[MAX_BOARD_SIZE], float (*symmetries_pi)[MAX_BOARD_SIZE], int* num_symmetries);
void chess_string_representation(const IGame* self, const int* board, char* str, int str_size);
void chess_display(const IGame* self, const int* board);

// Helper functions
__host__ __device__ bool is_check(const ChessBoard* board, int player);
__host__ __device__ bool is_checkmate(const ChessBoard* board, int player);
__host__ __device__ bool is_stalemate(const ChessBoard* board, int player);
__host__ __device__ bool is_insufficient_material(const ChessBoard* board);
__host__ __device__ bool is_fifty_move_rule(const ChessBoard* board);

// New helper functions
__host__ __device__ bool is_legal_move(const ChessBoard* board, int start, int end);
__host__ __device__ void make_move(ChessBoard* board, int start, int end);
__host__ __device__ bool can_castle_kingside(const ChessBoard* board, int player);
__host__ __device__ bool can_castle_queenside(const ChessBoard* board, int player);

typedef unsigned long long U64;
__host__ __device__ U64 get_attacks(int piece, int square, U64 occupancy);
__host__ __device__ U64 get_king_attacks(int square);
__host__ __device__ U64 get_pawn_attacks(int square, int color);
__host__ __device__ U64 get_knight_attacks(int square);
__host__ __device__ U64 get_sliding_attacks(int piece, int square, U64 occupancy);

// Zobrist hashing function declaration
__device__ __host__ unsigned long long compute_zobrist_hash(const ChessBoard* board);

// Zobrist random number table declarations
extern __device__ __constant__ unsigned long long ZOBRIST_PIECES[ZOBRIST_PIECE_TYPES][ZOBRIST_SQUARES];
extern __device__ __constant__ unsigned long long ZOBRIST_CASTLING[16];
extern __device__ __constant__ unsigned long long ZOBRIST_EN_PASSANT[8];
extern __device__ __constant__ unsigned long long ZOBRIST_SIDE_TO_MOVE;

// Create and destroy functions
ChessGame* create_chess_game();
void destroy_chess_game(ChessGame* game);

#endif // CHESS_H