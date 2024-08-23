// This is so messy tbh need to fix
#include "chess.cuh"

// Zobrist random number tables
__device__ __constant__ unsigned long long ZOBRIST_PIECES[ZOBRIST_PIECE_TYPES][ZOBRIST_SQUARES];
__device__ __constant__ unsigned long long ZOBRIST_CASTLING[16];
__device__ __constant__ unsigned long long ZOBRIST_EN_PASSANT[8];
__device__ __constant__ unsigned long long ZOBRIST_SIDE_TO_MOVE;

__device__ __constant__ U64 KING_ATTACKS[64];
__device__ __constant__ U64 KNIGHT_ATTACKS[64];
__device__ __constant__ U64 PAWN_ATTACKS[2][64];  // [color][square]

// Initialize the chess game state
__host__ __device__ void chess_init(IGame* self) {

    if (self == NULL) {
    // Handle error: self or board is NULL
        return;
    }
    // Initialize pieces
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        board->pieces[i] = EMPTY;
    }

    // Set up initial piece positions
    board->pieces[0] = board->pieces[7] = ROOK * WHITE;
    board->pieces[1] = board->pieces[6] = KNIGHT * WHITE;
    board->pieces[2] = board->pieces[5] = BISHOP * WHITE;
    board->pieces[3] = QUEEN * WHITE;
    board->pieces[4] = KING * WHITE;

    for (int i = 8; i < 16; i++) {
        board->pieces[i] = PAWN * WHITE;
    }

    for (int i = 48; i < 56; i++) {
        board->pieces[i] = PAWN * BLACK;
    }

    board->pieces[56] = board->pieces[63] = ROOK * BLACK;
    board->pieces[57] = board->pieces[62] = KNIGHT * BLACK;
    board->pieces[58] = board->pieces[61] = BISHOP * BLACK;
    board->pieces[59] = QUEEN * BLACK;
    board->pieces[60] = KING * BLACK;

    // Initialize other game state variables
    board->player = WHITE;
    board->castling_rights[0][0] = board->castling_rights[0][1] = true;
    board->castling_rights[1][0] = board->castling_rights[1][1] = true;
    board->en_passant_target = -1;
    board->halfmove_clock = 0;
    board->fullmove_number = 1;
    
    board->history_count = 0;
    for (int i = 0; i < MAX_HISTORY; i++) {
        board->position_history[i] = 0ULL;
    }

    // Compute and store the initial position hash
    board->position_history[0] = compute_zobrist_hash(board);
    board->history_count = 1;

}

// Set up the initial chess board configuration
__host__ __device__ void chess_get_init_board(const IGame* self, int* board) {
    if (self == NULL || board == NULL) {
        // Handle error: self or board is NULL
        return;
    }

    const ChessGame* chess = (const ChessGame*)self;
    if (chess == NULL) {
        // Handle error: invalid cast
        return;
    }

    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        board[i] = chess->board.pieces[i];
    }
}

// Return the dimensions of the chess board (8x8)
__host__ __device__ void chess_get_board_size(const IGame* self, int* rows, int* cols) {
    if (rows == NULL || cols == NULL) {
        // Handle error: rows or cols is NULL
        return;
    }

    *rows = 8;
    *cols = 8;
}

// Return the total number of possible moves in chess (e.g., 64 * 73 for all possible start and end squares including promotions)
__host__ __device__ int chess_get_action_size(const IGame* self) {
    // In chess, there are 64 possible starting squares and 73 possible target squares
    // (including the 9 possible underpromotion moves for pawns)
    return 64 * 73;
}

__host__ __device__ void chess_get_next_state(const IGame* self, const int* board, int player, int action, int* next_board, int* next_player) {
    if (self == NULL || board == NULL || next_board == NULL || next_player == NULL) {
        // Handle error: invalid input
        return;
    }
    
    const ChessGame* chess = (const ChessGame*)self;
    if (chess == NULL) {
        // Handle error: invalid cast
        return;
    }
    
    // Copy the current board state to the next board
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        next_board[i] = board[i];
    }
    
    // Create a temporary ChessBoard to work with
    ChessBoard temp_board = chess->board;
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        temp_board.pieces[i] = board[i];
    }
    temp_board.player = player;
    
    // Decode the action into start and end positions
    int start = action / 73;
    int end = action % 73;
    int promotion = QUEEN;  // Default promotion piece

    // Handle promotion moves
    if (end >= 64) {
        promotion = end - 64 + KNIGHT;  // KNIGHT, BISHOP, ROOK, QUEEN
        end = start / 8 == 1 ? start + 8 : start - 8;  // Move to the last rank
    }
    
    // Get the moving piece and capture piece (if any)
    int moving_piece = temp_board.pieces[start];
    int capture_piece = temp_board.pieces[end];
    
    // Apply the basic move
    temp_board.pieces[start] = EMPTY;
    temp_board.pieces[end] = moving_piece;
    
    // Handle special moves
    int piece_type = abs(moving_piece);
    int rank_start = start / 8;
    int file_start = start % 8;
    int rank_end = end / 8;
    int file_end = end % 8;
    
    // Pawn promotion
    if (piece_type == PAWN && (rank_end == 0 || rank_end == 7)) {
        temp_board.pieces[end] = player * promotion;
    }
    
    // En passant capture
    if (piece_type == PAWN && end == temp_board.en_passant_target) {
        int captured_pawn_pos = end + (player == WHITE ? -8 : 8);
        temp_board.pieces[captured_pawn_pos] = EMPTY;
    }
    
    // Update en passant target
    if (piece_type == PAWN && abs(rank_end - rank_start) == 2) {
        temp_board.en_passant_target = (start + end) / 2;
    } else {
        temp_board.en_passant_target = -1; // Reset en passant target
    }
    
    // Castling
    if (piece_type == KING && abs(file_end - file_start) == 2) {
        int rook_start, rook_end;
        if (file_end > file_start) { // Kingside castling
            rook_start = start + 3;
            rook_end = start + 1;
        } else { // Queenside castling
            rook_start = start - 4;
            rook_end = start - 1;
        }
        temp_board.pieces[rook_end] = temp_board.pieces[rook_start];
        temp_board.pieces[rook_start] = EMPTY;
    }
    
    // Update castling rights
    int player_index = player == WHITE ? 0 : 1;
    if (piece_type == KING) {
        temp_board.castling_rights[player_index][0] = false;
        temp_board.castling_rights[player_index][1] = false;
    } else if (piece_type == ROOK) {
        if (start == 0 || start == 56) // Queenside rook
            temp_board.castling_rights[player_index][1] = false;
        else if (start == 7 || start == 63) // Kingside rook
            temp_board.castling_rights[player_index][0] = false;
    }
    
    // Update halfmove clock
    if (piece_type == PAWN || capture_piece != EMPTY)
        temp_board.halfmove_clock = 0;
    else
        temp_board.halfmove_clock++;
    
    // Update fullmove number
    if (player == BLACK)
        temp_board.fullmove_number++;
    
    // Update the player
    temp_board.player = -player;
    *next_player = -player;
    
    // Copy the updated chess board back to the next_board array
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        next_board[i] = temp_board.pieces[i];
    }
}

__host__ __device__ void chess_get_valid_moves(const IGame* self, const int* board, int player, bool* valid_moves) {
    const ChessGame* chess_game = (const ChessGame*)self;
    ChessBoard temp_board;

    // Copy the board state to a temporary ChessBoard
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        temp_board.pieces[i] = board[i];
    }
    temp_board.player = player;
    temp_board.en_passant_target = chess_game->board.en_passant_target;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            temp_board.castling_rights[i][j] = chess_game->board.castling_rights[i][j];
        }
    }

    // Initialize all moves as invalid
    for (int i = 0; i < CHESS_BOARD_SIZE * 73; i++) {
        valid_moves[i] = false;
    }

    // Iterate through all pieces of the current player
    for (int start = 0; start < CHESS_BOARD_SIZE; start++) {
        int piece = temp_board.pieces[start];
        if (piece * player <= 0) continue; // Skip empty squares and opponent's pieces

        // Generate legal moves for the current piece
        for (int end = 0; end < CHESS_BOARD_SIZE; end++) {
            if (is_legal_move(&temp_board, start, end)) {
                // Make the move
                ChessBoard next_board = temp_board;
                make_move(&next_board, start, end);

                // Check if the move leaves the player in check
                if (!is_check(&next_board, player)) {
                    valid_moves[start * 73 + end] = true;

                    // Handle pawn promotion
                    if (abs(piece) == PAWN && (end / 8 == 0 || end / 8 == 7)) {
                        valid_moves[start * 73 + end] = false; // Disable default move
                        for (int promotion = KNIGHT; promotion <= QUEEN; promotion++) {
                            valid_moves[start * 73 + (64 + promotion - KNIGHT)] = true;
                        }
                    }
                }
            }
        }

        // Handle castling
        if (abs(piece) == KING) {
            if (can_castle_kingside(&temp_board, player)) {
                int kingside_end = start + 2;
                if (!is_check(&temp_board, player) && 
                    !is_square_attacked(&temp_board, start + 1, -player) &&
                    !is_square_attacked(&temp_board, kingside_end, -player)) {
                    valid_moves[start * 73 + kingside_end] = true;
                }
            }
            if (can_castle_queenside(&temp_board, player)) {
                int queenside_end = start - 2;
                if (!is_check(&temp_board, player) && 
                    !is_square_attacked(&temp_board, start - 1, -player) &&
                    !is_square_attacked(&temp_board, queenside_end, -player)) {
                    valid_moves[start * 73 + queenside_end] = true;
                }
            }
        }
    }
}

__host__ __device__ int chess_get_game_ended(const IGame* self, const int* board, int player) {
    const ChessGame* chess_game = (const ChessGame*)self;
    ChessBoard temp_board;

    // Copy the input board to the temporary board
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        temp_board.pieces[i] = board[i];
    }
    temp_board.player = player;
    temp_board.en_passant_target = chess_game->board.en_passant_target;
    temp_board.halfmove_clock = chess_game->board.halfmove_clock;
    temp_board.fullmove_number = chess_game->board.fullmove_number;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            temp_board.castling_rights[i][j] = chess_game->board.castling_rights[i][j];
        }
    }

    // Check for checkmate
    if (is_checkmate(&temp_board, player)) {
        return -player; // Return the opposite of the current player (winner)
    }

    // Check for stalemate
    if (is_stalemate(&temp_board, player)) {
        return 1e-4; // Draw
    }

    // Check for insufficient material
    if (is_insufficient_material(&temp_board)) {
        return 1e-4; // Draw
    }

    // Check for fifty-move rule
    if (is_fifty_move_rule(&temp_board)) {
        return 1e-4; // Draw
    }

    // Game hasn't ended
    return 0;
}
__host__ __device__ void chess_get_canonical_form(const IGame* self, const int* board, int player, int* canonical_board) {
    if (player == WHITE) {
        // If player is White, simply copy the board
        for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
            canonical_board[i] = board[i];
        }
    } else {
        // If player is Black, flip the board vertically and negate the pieces
        for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
            int flipped_index = CHESS_BOARD_SIZE - 1 - i;
            canonical_board[i] = -board[flipped_index];
        }
    }
}

// Heuristic evaluation of the board state (optional, can return 0 if not implemented)
__host__ __device__ float chess_evaluate(const IGame* self, const int* board, int player) {
    float score = 0;
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        int piece = board[i];
        switch (abs(piece)) {
            case PAWN:   score += 1.0f * (piece > 0 ? 1 : -1); break;
            case KNIGHT: score += 3.0f * (piece > 0 ? 1 : -1); break;
            case BISHOP: score += 3.0f * (piece > 0 ? 1 : -1); break;
            case ROOK:   score += 5.0f * (piece > 0 ? 1 : -1); break;
            case QUEEN:  score += 9.0f * (piece > 0 ? 1 : -1); break;
        }
    }
    return score * player;
}
// Generate all symmetries of the board (rotations and reflections)
void chess_get_symmetries(const IGame* self, const int* board, const float* pi, int (*symmetries)[CHESS_BOARD_SIZE], float (*symmetries_pi)[CHESS_BOARD_SIZE * 73], int* num_symmetries) {
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        symmetries[0][i] = board[i];
    }
    for (int i = 0; i < CHESS_BOARD_SIZE * 73; i++) {
        symmetries_pi[0][i] = pi[i];
    }
    *num_symmetries = 1;
}

// Convert the board state to a string representation
void chess_string_representation(const IGame* self, const int* board, char* str, int str_size) {
    const char* pieces = " PNBRQK pnbrqk";
    int idx = 0;
    for (int rank = 7; rank >= 0; rank--) {
        idx += snprintf(str + idx, str_size - idx, "%d ", rank + 1);
        for (int file = 0; file < 8; file++) {
            int piece = board[rank * 8 + file];
            str[idx++] = pieces[piece < 0 ? (6 - piece) : piece];
            if (idx >= str_size - 1) goto end;
            if (file < 7) str[idx++] = ' ';
            if (idx >= str_size - 1) goto end;
        }
        if (rank > 0) {
            str[idx++] = '\n';
            if (idx >= str_size - 1) goto end;
        }
    }
    idx += snprintf(str + idx, str_size - idx, "\n  a b c d e f g h");
end:
    str[idx] = '\0';
}

// Display the current board state (e.g., print to console)
void chess_display(const IGame* self, const int* board) {
    char str[256];
    chess_string_representation(self, board, str, sizeof(str));
    printf("%s\n", str);
}

// Create a new chess game instance
ChessGame* create_chess_game() {
    ChessGame* game = (ChessGame*)malloc(sizeof(ChessGame));
    if (!game) {
        fprintf(stderr, "Failed to allocate memory for ChessGame\n");
        return NULL;
    }
    game->base.init = chess_init;
    game->base.get_init_board = chess_get_init_board;
    game->base.get_board_size = chess_get_board_size;
    game->base.get_action_size = chess_get_action_size;
    game->base.get_next_state = chess_get_next_state;
    game->base.get_valid_moves = chess_get_valid_moves;
    game->base.get_game_ended = chess_get_game_ended;
    game->base.get_canonical_form = chess_get_canonical_form;
    game->base.evaluate = chess_evaluate;
    chess_init(&game->base);
    return game;
}

// Free resources associated with a chess game instance
void destroy_chess_game(ChessGame* game) {
    if (game) {
        free(game);
    }
}

/***************************************************
 * HELPERS
 **************************************************/

/__host__ __device__ bool is_check(const ChessBoard* board, int player) {
    U64 king_bb = 0, occ = 0, attacks = 0;
    int king_square = -1;

    // Set up bitboards
    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        if (board->pieces[i] != EMPTY) {
            occ |= 1ULL << i;
            if (board->pieces[i] == player * KING) {
                king_bb = 1ULL << i;
                king_square = i;
            }
        }
    }

    if (king_square == -1) return false;  // No king found, shouldn't happen in a valid position

    // Check for pawn attacks
    attacks |= get_pawn_attacks(king_square, player);

    // Check for knight attacks
    attacks |= get_knight_attacks(king_square);

    // Check for sliding piece attacks (bishop, rook, queen)
    attacks |= get_sliding_attacks(BISHOP, king_square, occ);
    attacks |= get_sliding_attacks(ROOK, king_square, occ);

    // Check if any enemy piece is on an attacking square
    while (attacks) {
        int sq = LSB(attacks);
        if (board->pieces[sq] * player < 0) {  // Enemy piece
            int piece_type = abs(board->pieces[sq]);
            if (piece_type == PAWN || piece_type == KNIGHT ||
                (piece_type == BISHOP && (get_sliding_attacks(BISHOP, sq, occ) & king_bb)) ||
                (piece_type == ROOK && (get_sliding_attacks(ROOK, sq, occ) & king_bb)) ||
                (piece_type == QUEEN && ((get_sliding_attacks(BISHOP, sq, occ) | get_sliding_attacks(ROOK, sq, occ)) & king_bb))) {
                return true;
            }
        }
        POP_LSB(attacks);
    }

    return false;
}

__host__ __device__ bool is_checkmate(const ChessBoard* board, int player) {
    if (!is_check(board, player)) return false;

    // Try all possible moves for the player
    for (int from = 0; from < CHESS_BOARD_SIZE; from++) {
        if (board->pieces[from] * player <= 0) continue;  // Not player's piece

        U64 moves = get_attacks(board->pieces[from], from, 0);  // Pseudo-legal moves
        while (moves) {
            int to = LSB(moves);
            
            // Make the move
            ChessBoard new_board = *board;
            new_board.pieces[to] = new_board.pieces[from];
            new_board.pieces[from] = EMPTY;

            // If this move gets out of check, it's not checkmate
            if (!is_check(&new_board, player)) return false;

            POP_LSB(moves);
        }
    }

    return true;  // No legal moves found
}

__host__ __device__ bool is_stalemate(const ChessBoard* board, int player) {
    if (is_check(board, player)) return false;

    // Try all possible moves for the player
    for (int from = 0; from < CHESS_BOARD_SIZE; from++) {
        if (board->pieces[from] * player <= 0) continue;  // Not player's piece

        U64 moves = get_attacks(board->pieces[from], from, 0);  // Pseudo-legal moves
        while (moves) {
            int to = LSB(moves);
            
            // Make the move
            ChessBoard new_board = *board;
            new_board.pieces[to] = new_board.pieces[from];
            new_board.pieces[from] = EMPTY;

            // If this move doesn't result in check, it's a legal move
            if (!is_check(&new_board, player)) return false;

            POP_LSB(moves);
        }
    }

    return true;  // No legal moves found
}

__host__ __device__ bool is_insufficient_material(const ChessBoard* board) {
    int white_knights = 0, white_bishops = 0, white_bishop_color = -1;
    int black_knights = 0, black_bishops = 0, black_bishop_color = -1;
    int total_pieces = 0;

    for (int i = 0; i < CHESS_BOARD_SIZE; i++) {
        int piece = board->pieces[i];
        if (piece != EMPTY) {
            total_pieces++;
            switch (abs(piece)) {
                case PAWN:
                case ROOK:
                case QUEEN:
                    return false;  // Sufficient material
                case KNIGHT:
                    if (piece > 0) white_knights++;
                    else black_knights++;
                    break;
                case BISHOP:
                    if (piece > 0) {
                        white_bishops++;
                        white_bishop_color = (white_bishop_color == -1) ? (i % 2) : white_bishop_color;
                    } else {
                        black_bishops++;
                        black_bishop_color = (black_bishop_color == -1) ? (i % 2) : black_bishop_color;
                    }
                    break;
            }
        }
    }

    // King vs. King
    if (total_pieces == 2) return true;

    // King and Bishop vs. King or King and Knight vs. King
    if (total_pieces == 3 && (white_bishops == 1 || black_bishops == 1 || white_knights == 1 || black_knights == 1)) return true;

    // King and two Knights vs. King
    if (total_pieces == 4 && ((white_knights == 2 && black_bishops == 0 && black_knights == 0) || 
                              (black_knights == 2 && white_bishops == 0 && white_knights == 0))) return true;

    // Both sides have a single bishop, and they are on the same color
    if (white_bishops == 1 && black_bishops == 1 && white_bishop_color == black_bishop_color && 
        white_knights == 0 && black_knights == 0) return true;

    return false;
}

__host__ __device__ bool is_threefold_repetition(const ChessBoard* board) {
    if (board->history_count < 4) return false;  // Need at least 4 moves for a threefold repetition

    unsigned long long current_hash = compute_zobrist_hash(board);
    int repetition_count = 1;  // Current position counts as 1

    for (int i = board->history_count - 1; i >= 0; i--) {
        if (board->position_history[i] == current_hash) {
            repetition_count++;
            if (repetition_count >= 3) return true;
        }
    }

    return false;
}

// Initialize Zobrist tables (call this function once at the start of your program)
void initialize_zobrist() {
    unsigned long long host_pieces[ZOBRIST_PIECE_TYPES][ZOBRIST_SQUARES];
    unsigned long long host_castling[16];
    unsigned long long host_en_passant[8];
    unsigned long long host_side_to_move;

    // Initialize random number generator
    curandState_t state;
    curand_init(1234, 0, 0, &state);

    // Generate random numbers for pieces
    for (int i = 0; i < ZOBRIST_PIECE_TYPES; i++) {
        for (int j = 0; j < ZOBRIST_SQUARES; j++) {
            host_pieces[i][j] = curand(&state);
        }
    }

    // Generate random numbers for castling rights
    for (int i = 0; i < 16; i++) {
        host_castling[i] = curand(&state);
    }

    // Generate random numbers for en passant files
    for (int i = 0; i < 8; i++) {
        host_en_passant[i] = curand(&state);
    }

    // Generate random number for side to move
    host_side_to_move = curand(&state);

    // Copy to device constant memory
    cudaMemcpyToSymbol(ZOBRIST_PIECES, host_pieces, sizeof(host_pieces));
    cudaMemcpyToSymbol(ZOBRIST_CASTLING, host_castling, sizeof(host_castling));
    cudaMemcpyToSymbol(ZOBRIST_EN_PASSANT, host_en_passant, sizeof(host_en_passant));
    cudaMemcpyToSymbol(ZOBRIST_SIDE_TO_MOVE, &host_side_to_move, sizeof(host_side_to_move));
}

// Zobrist hash computation function
__device__ __host__ unsigned long long compute_zobrist_hash(const ChessBoard* board) {
    unsigned long long hash = 0;

    // Hash pieces
    for (int sq = 0; sq < 64; sq++) {
        int piece = board->pieces[sq];
        if (piece != EMPTY) {
            int piece_index = (abs(piece) - 1) * 2 + (piece > 0 ? 0 : 1);
            hash ^= ZOBRIST_PIECES[piece_index][sq];
        }
    }

    // Hash castling rights
    int castling_index = (board->castling_rights[WHITE][0] << 3) |
                         (board->castling_rights[WHITE][1] << 2) |
                         (board->castling_rights[BLACK][0] << 1) |
                         (board->castling_rights[BLACK][1]);
    hash ^= ZOBRIST_CASTLING[castling_index];

    // Hash en passant
    if (board->en_passant_target != -1) {
        int file = board->en_passant_target % 8;
        hash ^= ZOBRIST_EN_PASSANT[file];
    }

    // Hash side to move
    if (board->player == BLACK) {
        hash ^= ZOBRIST_SIDE_TO_MOVE;
    }

    return hash;
}

// Check if the fifty-move rule applies
__host__ __device__ bool is_fifty_move_rule(const ChessBoard* board) {
    // The fifty-move rule states that a player can claim a draw 
    // if no capture has been made and no pawn has been moved in 
    // the last fifty moves (by both players).
    
    // In chess, the halfmove clock keeps track of the number of 
    // halfmoves (or plies) since the last pawn move or capture.
    // It's reset to 0 after a capture or a pawn move.
    
    // The fifty-move rule is invoked when the halfmove clock 
    // reaches 100 (50 full moves, which is 100 halfmoves).
    
    return board->halfmove_clock >= 100;
}
__host__ __device__ bool is_legal_move(const ChessBoard* board, int start, int end) {
    int piece = board->pieces[start];
    int player = board->player;
    
    // Check if the piece belongs to the current player
    if (piece * player <= 0) return false;
    
    // Check if the destination square is occupied by a friendly piece
    if (board->pieces[end] * player > 0) return false;
    
    int piece_type = abs(piece);
    int start_rank = start / 8;
    int start_file = start % 8;
    int end_rank = end / 8;
    int end_file = end % 8;
    
    // Check piece-specific movement rules
    switch (piece_type) {
        case PAWN:
            // Pawn move
            if (player == WHITE) {
                if (start_rank == 1 && end_rank == 3 && start_file == end_file && board->pieces[start + 8] == EMPTY)
                    return true; // Double step from initial position
                if (end_rank == start_rank + 1) {
                    if (start_file == end_file && board->pieces[end] == EMPTY)
                        return true; // Single step forward
                    if (abs(end_file - start_file) == 1 && (board->pieces[end] < 0 || end == board->en_passant_target))
                        return true; // Capture or en passant
                }
            } else { // BLACK
                if (start_rank == 6 && end_rank == 4 && start_file == end_file && board->pieces[start - 8] == EMPTY)
                    return true; // Double step from initial position
                if (end_rank == start_rank - 1) {
                    if (start_file == end_file && board->pieces[end] == EMPTY)
                        return true; // Single step forward
                    if (abs(end_file - start_file) == 1 && (board->pieces[end] > 0 || end == board->en_passant_target))
                        return true; // Capture or en passant
                }
            }
            break;
        
        case KNIGHT:
            if ((abs(end_rank - start_rank) == 2 && abs(end_file - start_file) == 1) ||
                (abs(end_rank - start_rank) == 1 && abs(end_file - start_file) == 2))
                return true;
            break;
        
        case BISHOP:
            if (abs(end_rank - start_rank) == abs(end_file - start_file)) {
                int rank_step = (end_rank > start_rank) ? 1 : -1;
                int file_step = (end_file > start_file) ? 1 : -1;
                for (int r = start_rank + rank_step, f = start_file + file_step; 
                     r != end_rank; r += rank_step, f += file_step) {
                    if (board->pieces[r * 8 + f] != EMPTY)
                        return false; // Path is blocked
                }
                return true;
            }
            break;
        
        case ROOK:
            if (start_rank == end_rank || start_file == end_file) {
                int step = (start_rank == end_rank) ? 
                           ((end_file > start_file) ? 1 : -1) : 
                           ((end_rank > start_rank) ? 8 : -8);
                for (int sq = start + step; sq != end; sq += step) {
                    if (board->pieces[sq] != EMPTY)
                        return false; // Path is blocked
                }
                return true;
            }
            break;
        
        case QUEEN:
            if (start_rank == end_rank || start_file == end_file || 
                abs(end_rank - start_rank) == abs(end_file - start_file)) {
                int rank_step = (end_rank == start_rank) ? 0 : ((end_rank > start_rank) ? 1 : -1);
                int file_step = (end_file == start_file) ? 0 : ((end_file > start_file) ? 1 : -1);
                for (int r = start_rank + rank_step, f = start_file + file_step; 
                     r != end_rank || f != end_file; r += rank_step, f += file_step) {
                    if (board->pieces[r * 8 + f] != EMPTY)
                        return false; // Path is blocked
                }
                return true;
            }
            break;
        
        case KING:
            if (abs(end_rank - start_rank) <= 1 && abs(end_file - start_file) <= 1)
                return true;
            // Check castling
            if (start_rank == end_rank && abs(end_file - start_file) == 2) {
                bool kingside = (end_file > start_file);
                return (kingside ? can_castle_kingside(board, player) : can_castle_queenside(board, player));
            }
            break;
    }
    
    return false;
}


__host__ __device__ void make_move(ChessBoard* board, int start, int end) {
    int moving_piece = board->pieces[start];
    int captured_piece = board->pieces[end];
    int player = board->player;

    // Basic move
    board->pieces[end] = moving_piece;
    board->pieces[start] = EMPTY;

    // Handle special moves
    int piece_type = abs(moving_piece);
    int start_rank = start / 8;
    int start_file = start % 8;
    int end_rank = end / 8;
    int end_file = end % 8;

    // Pawn promotion
    if (piece_type == PAWN && (end_rank == 0 || end_rank == 7)) {
        // Assuming promotion to Queen by default
        board->pieces[end] = QUEEN * player;
    }

    // En passant capture
    if (piece_type == PAWN && end == board->en_passant_target) {
        int captured_pawn_pos = end + (player == WHITE ? -8 : 8);
        board->pieces[captured_pawn_pos] = EMPTY;
    }

    // Update en passant target
    if (piece_type == PAWN && abs(end_rank - start_rank) == 2) {
        board->en_passant_target = (start + end) / 2;
    } else {
        board->en_passant_target = -1;
    }

    // Castling
    if (piece_type == KING && abs(end_file - start_file) == 2) {
        int rook_start, rook_end;
        if (end_file > start_file) { // Kingside castling
            rook_start = start + 3;
            rook_end = start + 1;
        } else { // Queenside castling
            rook_start = start - 4;
            rook_end = start - 1;
        }
        board->pieces[rook_end] = board->pieces[rook_start];
        board->pieces[rook_start] = EMPTY;
    }

    // Update castling rights
    if (piece_type == KING) {
        board->castling_rights[player == WHITE ? 0 : 1][0] = false;
        board->castling_rights[player == WHITE ? 0 : 1][1] = false;
    } else if (piece_type == ROOK) {
        if (start == 0 || start == 56) // Queenside rook
            board->castling_rights[player == WHITE ? 0 : 1][1] = false;
        else if (start == 7 || start == 63) // Kingside rook
            board->castling_rights[player == WHITE ? 0 : 1][0] = false;
    }

    // Update halfmove clock
    if (piece_type == PAWN || captured_piece != EMPTY)
        board->halfmove_clock = 0;
    else
        board->halfmove_clock++;

    // Update fullmove number
    if (player == BLACK)
        board->fullmove_number++;

    // Switch player
    board->player = -player;

    // Update position history
    unsigned long long new_hash = compute_zobrist_hash(board);
    int index = board->history_count % MAX_HISTORY;
    board->position_history[index] = new_hash;
    board->history_count++;
}
}


__host__ __device__ bool can_castle_kingside(const ChessBoard* board, int player) {
    int king_pos = player == WHITE ? 4 : 60;
    int rook_pos = player == WHITE ? 7 : 63;
    
    // Check if castling rights are still available
    if (!board->castling_rights[player == WHITE ? 0 : 1][0]) {
        return false;
    }
    
    // Check if the squares between the king and rook are empty
    for (int i = king_pos + 1; i < rook_pos; i++) {
        if (board->pieces[i] != EMPTY) {
            return false;
        }
    }
    
    // Check if the king is in check
    if (is_square_attacked(board, king_pos, -player)) {
        return false;
    }
    
    // Check if the squares the king passes through are attacked
    if (is_square_attacked(board, king_pos + 1, -player) ||
        is_square_attacked(board, king_pos + 2, -player)) {
        return false;
    }
    
    return true;
}

__host__ __device__ bool can_castle_queenside(const ChessBoard* board, int player) {
    int king_pos = player == WHITE ? 4 : 60;
    int rook_pos = player == WHITE ? 0 : 56;
    
    // Check if castling rights are still available
    if (!board->castling_rights[player == WHITE ? 0 : 1][1]) {
        return false;
    }
    
    // Check if the squares between the king and rook are empty
    for (int i = rook_pos + 1; i < king_pos; i++) {
        if (board->pieces[i] != EMPTY) {
            return false;
        }
    }
    
    // Check if the king is in check
    if (is_square_attacked(board, king_pos, -player)) {
        return false;
    }
    
    // Check if the squares the king passes through are attacked
    if (is_square_attacked(board, king_pos - 1, -player) ||
        is_square_attacked(board, king_pos - 2, -player)) {
        return false;
    }
    
    return true;
}

// Pre-computed lookup tables (should be initialized at program start)

__host__ __device__ U64 get_attacks(int piece, int square, U64 occupancy) {
    switch (abs(piece)) {
        case KING:
            return get_king_attacks(square);
        case PAWN:
            return get_pawn_attacks(square, piece > 0 ? WHITE : BLACK);
        case KNIGHT:
            return get_knight_attacks(square);
        case BISHOP:
            return get_sliding_attacks(BISHOP, square, occupancy);
        case ROOK:
            return get_sliding_attacks(ROOK, square, occupancy);
        case QUEEN:
            return get_sliding_attacks(BISHOP, square, occupancy) | 
                   get_sliding_attacks(ROOK, square, occupancy);
        default:
            return 0ULL;  // Empty or invalid piece
    }
}

__host__ __device__ U64 get_king_attacks(int square) {
    return KING_ATTACKS[square];
}

__host__ __device__ U64 get_pawn_attacks(int square, int color) {
    return PAWN_ATTACKS[color == WHITE ? 0 : 1][square];
}

__host__ __device__ U64 get_knight_attacks(int square) {
    return KNIGHT_ATTACKS[square];
}

__host__ __device__ U64 get_sliding_attacks(int piece, int square, U64 occupancy) {
    U64 attacks = 0ULL;
    int directions[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    int start_dir = (piece == BISHOP) ? 2 : 0;
    int end_dir = (piece == ROOK) ? 2 : 4;

    for (int dir = start_dir; dir < end_dir; dir++) {
        for (int i = 1; i < 8; i++) {
            int rank = square / 8 + i * directions[dir][0];
            int file = square % 8 + i * directions[dir][1];
            if (rank < 0 || rank >= 8 || file < 0 || file >= 8) break;
            
            U64 square_bb = 1ULL << (rank * 8 + file);
            attacks |= square_bb;
            
            if (occupancy & square_bb) break;
        }
        
        for (int i = 1; i < 8; i++) {
            int rank = square / 8 - i * directions[dir][0];
            int file = square % 8 - i * directions[dir][1];
            if (rank < 0 || rank >= 8 || file < 0 || file >= 8) break;
            
            U64 square_bb = 1ULL << (rank * 8 + file);
            attacks |= square_bb;
            
            if (occupancy & square_bb) break;
        }
    }

    return attacks;
}

// Helper function to initialize attack tables (call this at program start)
void initialize_attack_tables() {
    for (int sq = 0; sq < 64; sq++) {
        U64 bb = 1ULL << sq;
        int rank = sq / 8, file = sq % 8;

        // King attacks
        U64 king_att = ((bb << 1) & 0xfefefefefefefefeULL) | ((bb >> 1) & 0x7f7f7f7f7f7f7f7fULL);
        king_att |= ((bb << 7) & 0x7f7f7f7f7f7f7f7fULL) | ((bb >> 7) & 0xfefefefefefefefeULL);
        king_att |= (bb << 8) | (bb >> 8) | ((bb << 9) & 0xfefefefefefefefeULL) | ((bb >> 9) & 0x7f7f7f7f7f7f7f7fULL);
        KING_ATTACKS[sq] = king_att;

        // Knight attacks
        U64 knight_att = 0ULL;
        for (int i = 0; i < 8; i++) {
            int r = rank + ((i < 4) ? 2 : -2) * (((i & 1) == 0) ? 1 : -1);
            int f = file + ((i < 4) ? 1 : -1) * (((i & 1) == 0) ? 2 : -2);
            if (r >= 0 && r < 8 && f >= 0 && f < 8)
                knight_att |= 1ULL << (r * 8 + f);
        }
        KNIGHT_ATTACKS[sq] = knight_att;

        // Pawn attacks
        U64 white_pawn_att = ((bb << 7) & 0x7f7f7f7f7f7f7f7fULL) | ((bb << 9) & 0xfefefefefefefefeULL);
        U64 black_pawn_att = ((bb >> 7) & 0xfefefefefefefefeULL) | ((bb >> 9) & 0x7f7f7f7f7f7f7f7fULL);
        PAWN_ATTACKS[0][sq] = white_pawn_att;
        PAWN_ATTACKS[1][sq] = black_pawn_att;
    }
}

__host__ __device__ bool is_square_attacked(const ChessBoard* board, int square, int attacker) {
    // Check pawn attacks
    int pawn_direction = attacker == WHITE ? -1 : 1;
    int pawn_rank = square / 8 + pawn_direction;
    int pawn_file = square % 8;
    if (pawn_rank >= 0 && pawn_rank < 8) {
        if (pawn_file > 0 && board->pieces[pawn_rank * 8 + pawn_file - 1] == PAWN * attacker) return true;
        if (pawn_file < 7 && board->pieces[pawn_rank * 8 + pawn_file + 1] == PAWN * attacker) return true;
    }

    // Check knight attacks
    int knight_moves[8][2] = {{-2,-1}, {-2,1}, {-1,-2}, {-1,2}, {1,-2}, {1,2}, {2,-1}, {2,1}};
    for (int i = 0; i < 8; i++) {
        int rank = square / 8 + knight_moves[i][0];
        int file = square % 8 + knight_moves[i][1];
        if (rank >= 0 && rank < 8 && file >= 0 && file < 8) {
            if (board->pieces[rank * 8 + file] == KNIGHT * attacker) return true;
        }
    }

    // Check king attacks (needed for castling checks)
    int king_moves[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
    for (int i = 0; i < 8; i++) {
        int rank = square / 8 + king_moves[i][0];
        int file = square % 8 + king_moves[i][1];
        if (rank >= 0 && rank < 8 && file >= 0 && file < 8) {
            if (board->pieces[rank * 8 + file] == KING * attacker) return true;
        }
    }

    // Check sliding piece attacks (bishop, rook, queen)
    int directions[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
    for (int i = 0; i < 8; i++) {
        int rank = square / 8;
        int file = square % 8;
        while (true) {
            rank += directions[i][0];
            file += directions[i][1];
            if (rank < 0 || rank >= 8 || file < 0 || file >= 8) break;
            int piece = board->pieces[rank * 8 + file];
            if (piece != EMPTY) {
                if (piece * attacker > 0) {
                    if ((i < 4 && abs(piece) == ROOK) ||
                        (i >= 4 && abs(piece) == BISHOP) ||
                        abs(piece) == QUEEN) {
                        return true;
                    }
                }
                break;
            }
        }
    }

    return false;
}