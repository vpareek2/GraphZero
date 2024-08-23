#include "neural_network.h"
#include "resnet/resnet.cuh"
#include "gat/gat.cuh"
// Include headers for other network types as you implement them

INeuralNet* create_neural_net(const char* net_type, const IGame* game) {
    if (strcmp(net_type, "resnet") == 0) {
        return create_resnet_model(game);
    } else if (strcmp(net_type, "gat") == 0) {
        return create_gat_model(game);
    }
    // Add other network types here as you implement them
    return NULL;  // Unknown network type
}