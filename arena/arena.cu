#include "arena.cuh"
#include <stdio.h>
#include <stdlib.h>

Arena* create_arena(Player* player1, Player* player2, IGame* game) {
    Arena* arena = (Arena*)malloc(sizeof(Arena)); // Allocate memory for the arena
    if (arena == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for the arena\n");
        return NULL;
    }
    arena->player1 = player1; // Initialize player 1
    arena->player2 = player2; // Initialize player 2
    arena->game = game; // Initialize game instance
    arena->board_size = (int*)malloc(sizeof(int)); // Allocate memory for the board size
    if (arena->board_size == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for the board size\n");
        free(arena); // Free arena memory
        return NULL;
    }
    *arena->board_size = MAX_BOARD_SIZE; // Set the board size
    arena->board = (int*)malloc(sizeof(int) * *arena->board_size); // Allocate memory for the board
    if (arena->board == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for the board\n");
        free(arena->board_size); // Free board size memory
        free(arena); // Free arena memory
        return NULL;
    }
    return arena;
}

int play_game(Arena* arena, bool verbose) {
    if (arena == NULL) {
        fprintf(stderr, "Error: Arena is NULL\n");
        return 0; // Return error
    }
    if (arena->board == NULL) {
        fprintf(stderr, "Error: Board is NULL\n");
        return 0; // Return error
    }
    if (arena->board_size == NULL) {
        fprintf(stderr, "Error: Board size is NULL\n");
        return 0; // Return error
    }
    if (arena->game == NULL) {
        fprintf(stderr, "Error: Game is NULL\n");
        return 0; // Return error
    }
    int* board = arena->board;
    arena->game->get_init_board(arena->game, board);
    
    int current_player = 1;
    int game_result = 0;
    
    while (game_result == 0) {
        Player* current_player_obj = (current_player == 1) ? arena->player1 : arena->player2;
        if (current_player_obj == NULL) {
            fprintf(stderr, "Error: Current player object is NULL\n");
            return 0; // Return error
        }
        int action = current_player_obj->get_action(current_player_obj, board);
        
        if (!arena->game->is_valid_action(arena->game, board, action)) {
            fprintf(stderr, "Error: Invalid action %d\n", action);
            return 0; // Draw or handle error as appropriate
        }
        
        int next_board[*arena->board_size];
        int next_player;
        arena->game->get_next_state(arena->game, board, current_player, action, next_board, &next_player);
        
        memcpy(board, next_board, sizeof(int) * *arena->board_size);
        current_player = next_player;
        
        game_result = arena->game->get_game_ended(arena->game, board, current_player);
        
        if (verbose) {
            arena->game->display(arena->game, board);
        }
    }
    
    return game_result;
}

void play_games(Arena* arena, int num_games, int* wins, int* losses, int* draws) {
    if (arena == NULL) {
        fprintf(stderr, "Error: Arena is NULL\n");
        return; // Exit early if arena is NULL
    }
    *wins = 0;
    *losses = 0;
    *draws = 0;
    
    for (int i = 0; i < num_games; i++) {
        int result = play_game(arena, false);
        if (result == 1) (*wins)++;
        else if (result == -1) (*losses)++;
        else (*draws)++;
        
        // Swap players every other game
        if (i == num_games / 2 - 1) {
            Player* temp = arena->player1;
            arena->player1 = arena->player2;
            arena->player2 = temp;
        }
    }
}

void destroy_arena(Arena* arena) {
    if (arena == NULL) {
        return; // Exit early if arena is NULL
    }
    free(arena->board); // Free board memory
    free(arena->board_size); // Free board size memory
    free(arena); // Free arena memory
}