#ifndef CONFIG_H
#define CONFIG_H

#include <cjson/cJSON.h>
#include "../mcts/mcts.cuh"
#include "../networks/neural_network.h"
#include "../networks/gat/gat.cuh"
#include "../self_play/self_play.cuh"

typedef struct {
    struct {
        int max_size;
    } board;

    struct {
        int max_children;
        float c_puct;
        int num_simulations;
        int threads_per_block;
        int max_batch_size;
    } mcts;

    struct {
        int max_filename_length;
    } neural_network;

    struct {
        ModelConfig model_config;
    } gat;

    struct {
        int max_batch_size;
        int max_num_games;
        int max_game_length;
        int terminal_state;
        int max_filename_length;
        SelfPlayConfig config;
    } self_play;
} GlobalConfig;

GlobalConfig* load_config(const char* filename);
void free_config(GlobalConfig* config);

#endif // CONFIG_H