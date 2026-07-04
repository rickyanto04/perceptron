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
    Network *nn = create_network(num_layers, layer_sizes, learning_rate);

    printf("Architettura: %d strati (%d -> %d -> %d)\n",
           num_layers, layer_sizes[0], layer_sizes[1], layer_sizes[2]);

    printf("Rete allocata con successo nella memoria Heap!\n");
    printf("Verifica neurone casuale (Hidden Layer, ID 0): Bias = %f\n", nn->layers[1].neurons[0].bias);

    /*Creazione immagine fittizia tutta nera 0.0 e un pixel bianco 1.0*/
    double *dummy_image = (double*)malloc(784 * sizeof(double));
    for (int i = 0; i < 784; i++) {
        dummy_image[i] = 0.0;
    }
    dummy_image[400] = 1.0;

    printf("Esecuzione forward pass...\n");
    forward_pass(nn, dummy_image);

    printf("\nRisultati Output Layer (predizione per classi 0-9):\n");
    Layer *output_layer = &nn->layers[num_layers - 1];
    for (int i = 0; i < output_layer->num_neurons; i++) {
        printf("Neurone %d: probabilita' %.4f\n", i, output_layer->neurons[i].output);
    }

    free(dummy_image);
    free_network(nn);
    printf("Memoria deallocata correttamente, 0 leaks.\n");

    return 0;
}