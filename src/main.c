#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "neural_math.h"

int main() {

    srand((unsigned int)time(NULL));

    // architettura per il riconoscimento immagini(MNIST)
    // 784 pixel input -> 128 neuroni nascosti -> 10 neuroni output(classi 0-9)
    int layer_sizes[] = {784, 128, 10};
    int num_layers = 3;
    double learning_rate = 0.1;

    printf("--- Inizializzazione Rete Neurale ---\n");
    printf("Architettura: %d strati (%d -> %d -> %d)\n",
           num_layers, layer_sizes[0], layer_sizes[1], layer_sizes[2]);

    Network *nn = create_network(num_layers, layer_sizes, learning_rate);

    printf("Rete allocata con successo nella memoria Heap!\n");
    printf("Verifica neurone casuale (Hidden Layer, ID 0): Bias = %f\n", nn->layers[1].neurons[0].bias);

    free_network(nn);
    printf("Memoria deallocata correttamente, 0 leaks.\n");

    return 0;
}