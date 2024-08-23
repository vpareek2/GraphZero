#include "config.h"
#include <stdio.h>
#include <stdlib.h>

static void load_board_config(cJSON* json, GlobalConfig* config) {
    cJSON* board = cJSON_GetObjectItemCaseSensitive(json, "board");
    if (board) {
        config->board.max_size = cJSON_GetObjectItemCaseSensitive(board, "max_size")->valueint;
    }
}

static void load_mcts_config(cJSON* json, GlobalConfig* config) {
    cJSON* mcts = cJSON_GetObjectItemCaseSensitive(json, "mcts");
    if (mcts) {
        config->mcts.max_children = cJSON_GetObjectItemCaseSensitive(mcts, "max_children")->valueint;
        config->mcts.c_puct = cJSON_GetObjectItemCaseSensitive(mcts, "c_puct")->valuedouble;
        config->mcts.num_simulations = cJSON_GetObjectItemCaseSensitive(mcts, "num_simulations")->valueint;
        config->mcts.threads_per_block = cJSON_GetObjectItemCaseSensitive(mcts, "threads_per_block")->valueint;
        config->mcts.max_batch_size = cJSON_GetObjectItemCaseSensitive(mcts, "max_batch_size")->valueint;
    }
}

static void load_neural_network_config(cJSON* json, GlobalConfig* config) {
    cJSON* nn = cJSON_GetObjectItemCaseSensitive(json, "neural_network");
    if (nn) {
        config->neural_network.max_filename_length = cJSON_GetObjectItemCaseSensitive(nn, "max_filename_length")->valueint;
    }
}

static void load_gat_config(cJSON* json, GlobalConfig* config) {
    cJSON* gat = cJSON_GetObjectItemCaseSensitive(json, "gat");
    if (gat) {
        cJSON* model_config = cJSON_GetObjectItemCaseSensitive(gat, "model_config");
        if (model_config) {
            config->gat.model_config.input_features = cJSON_GetObjectItemCaseSensitive(model_config, "input_features")->valueint;
            config->gat.model_config.hidden_features = cJSON_GetObjectItemCaseSensitive(model_config, "hidden_features")->valueint;
            config->gat.model_config.output_features = cJSON_GetObjectItemCaseSensitive(model_config, "output_features")->valueint;
            config->gat.model_config.num_heads = cJSON_GetObjectItemCaseSensitive(model_config, "num_heads")->valueint;
            config->gat.model_config.num_layers = cJSON_GetObjectItemCaseSensitive(model_config, "num_layers")->valueint;
            config->gat.model_config.num_actions = cJSON_GetObjectItemCaseSensitive(model_config, "num_actions")->valueint;
            config->gat.model_config.max_nodes = cJSON_GetObjectItemCaseSensitive(model_config, "max_nodes")->valueint;
            config->gat.model_config.max_edges = cJSON_GetObjectItemCaseSensitive(model_config, "max_edges")->valueint;
            config->gat.model_config.learning_rate = cJSON_GetObjectItemCaseSensitive(model_config, "learning_rate")->valuedouble;
            config->gat.model_config.weight_decay = cJSON_GetObjectItemCaseSensitive(model_config, "weight_decay")->valuedouble;
            config->gat.model_config.dropout = cJSON_GetObjectItemCaseSensitive(model_config, "dropout")->valuedouble;
            config->gat.model_config.alpha = cJSON_GetObjectItemCaseSensitive(model_config, "alpha")->valuedouble;
            config->gat.model_config.batch_size = cJSON_GetObjectItemCaseSensitive(model_config, "batch_size")->valueint;
            config->gat.model_config.epochs = cJSON_GetObjectItemCaseSensitive(model_config, "epochs")->valueint;
        }
    }
}

static void load_self_play_config(cJSON* json, GlobalConfig* config) {
    cJSON* self_play = cJSON_GetObjectItemCaseSensitive(json, "self_play");
    if (self_play) {
        config->self_play.max_batch_size = cJSON_GetObjectItemCaseSensitive(self_play, "max_batch_size")->valueint;
        config->self_play.max_num_games = cJSON_GetObjectItemCaseSensitive(self_play, "max_num_games")->valueint;
        config->self_play.max_game_length = cJSON_GetObjectItemCaseSensitive(self_play, "max_game_length")->valueint;
        config->self_play.terminal_state = cJSON_GetObjectItemCaseSensitive(self_play, "terminal_state")->valueint;
        config->self_play.max_filename_length = cJSON_GetObjectItemCaseSensitive(self_play, "max_filename_length")->valueint;

        cJSON* sp_config = cJSON_GetObjectItemCaseSensitive(self_play, "config");
        if (sp_config) {
            config->self_play.config.numIters = cJSON_GetObjectItemCaseSensitive(sp_config, "numIters")->valueint;
            config->self_play.config.numEps = cJSON_GetObjectItemCaseSensitive(sp_config, "numEps")->valueint;
            config->self_play.config.numGames = cJSON_GetObjectItemCaseSensitive(sp_config, "numGames")->valueint;
            config->self_play.config.batchSize = cJSON_GetObjectItemCaseSensitive(sp_config, "batchSize")->valueint;
            config->self_play.config.numMCTSSims = cJSON_GetObjectItemCaseSensitive(sp_config, "numMCTSSims")->valueint;
            config->self_play.config.tempThreshold = cJSON_GetObjectItemCaseSensitive(sp_config, "tempThreshold")->valuedouble;
            config->self_play.config.updateThreshold = cJSON_GetObjectItemCaseSensitive(sp_config, "updateThreshold")->valuedouble;
            config->self_play.config.maxlenOfQueue = cJSON_GetObjectItemCaseSensitive(sp_config, "maxlenOfQueue")->valueint;
            config->self_play.config.numItersForTrainExamplesHistory = cJSON_GetObjectItemCaseSensitive(sp_config, "numItersForTrainExamplesHistory")->valueint;
            config->self_play.config.arenaCompare = cJSON_GetObjectItemCaseSensitive(sp_config, "arenaCompare")->valueint;
            strncpy(config->self_play.config.checkpoint, cJSON_GetObjectItemCaseSensitive(sp_config, "checkpoint")->valuestring, MAX_FILENAME_LENGTH - 1);
        }
    }
}

GlobalConfig* load_config(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open config file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* json_string = (char*)malloc(file_size + 1);
    fread(json_string, 1, file_size, file);
    fclose(file);
    json_string[file_size] = '\0';

    cJSON* json = cJSON_Parse(json_string);
    free(json_string);

    if (!json) {
        fprintf(stderr, "Error: Failed to parse JSON\n");
        return NULL;
    }

    GlobalConfig* config = (GlobalConfig*)malloc(sizeof(GlobalConfig));
    if (!config) {
        fprintf(stderr, "Error: Failed to allocate memory for config\n");
        cJSON_Delete(json);
        return NULL;
    }

    load_board_config(json, config);
    load_mcts_config(json, config);
    load_neural_network_config(json, config);
    load_gat_config(json, config);
    load_self_play_config(json, config);

    cJSON_Delete(json);
    return config;
}

void free_config(GlobalConfig* config) {
    if (config) {
        free(config);
    }
}