#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "neural_math.h"

double* dummy_4_factory() {
/*--- Creazione immagine fittizia che rappresenta il numero 4 ---*/
    double *dummy_image = (double*)malloc(784 * sizeof(double));
    if (dummy_image == NULL) {
        exit(1);
    }
    for (int i = 0; i < 784; i++) {
        dummy_image[i] = 0.0;
    }

    // Barra verticale sinistra
    for (int r = 6; r <= 15; r++){
        dummy_image[r * 28 + 10] = 1.0;
    }

    // Barra verticale destra
    for (int r = 4; r <= 22; r++){
        dummy_image[r * 28 + 17] = 1.0;
    }

    // Barra orizzontale centrale
    for (int c = 10; c <= 17; c++){
        dummy_image[15 * 28 + c] = 1.0;
    }

    return dummy_image;
}

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

    double* dummy_image_4 = dummy_4_factory();

    // Creazione Dato Target (One-Hot Encoding per la classe 4)
    double *target = (double*)malloc(10 * sizeof(double));
    for (int i = 0; i < 10; i++) target[i] = 0.0;
    target[4] = 1.0; // Vogliamo che il neurone con indice 4 si attivi

    printf("Inizio addestramento su singolo campione (1000 epoche)...\n");

    // ciclo di training
    int epochs = 1000;
    for (int e = 0; e < epochs; e++) {
        forward_pass(nn, dummy_image_4);
        backward_pass(nn, target);
        update_weights(nn);
    }

    printf("Esecuzione forward pass finale per verifica...\n");
    forward_pass(nn, dummy_image_4);

    printf("\nRisultati Output Layer POST-Addestramento (Target = Classe 4):\n");
    Layer *output_layer = &nn->layers[num_layers - 1];
    for (int i = 0; i < output_layer->num_neurons; i++) {
        // Se tutto funziona, il Neurone 4 avrà un valore > 0.95
        printf("Neurone %d: probabilita' %.4f\n", i, output_layer->neurons[i].output);
    }

    free(dummy_image_4);
    free(target);
    free_network(nn);
    printf("Memoria deallocata correttamente, 0 leaks.\n");

    return 0;
}