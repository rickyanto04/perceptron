#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "neural_math.h"

double* dummy_4_shifted_factory() {
    /*--- Creazione immagine alterata: barra destra spostata di 1 pixel ---*/
    double *dummy_image = (double*)malloc(784 * sizeof(double));
    if (dummy_image == NULL) {
        fprintf(stderr, "Errore di allocazione per l'immagine sfasata.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < 784; i++) {
        dummy_image[i] = 0.0;
    }

    // Barra verticale sinistra (Intatta)
    for (int r = 6; r <= 15; r++){
        dummy_image[r * 28 + 10] = 1.0;
    }

    // Barra verticale destra (SPOSTATA dalla colonna 17 alla 18)
    for (int r = 4; r <= 22; r++){
        dummy_image[r * 28 + 18] = 1.0;
    }

    // Barra orizzontale centrale (Allungata di 1 pixel per raccordare)
    for (int c = 10; c <= 18; c++){
        dummy_image[15 * 28 + c] = 1.0;
    }

    return dummy_image;
}

double* dummy_4_factory() {
/*--- Creazione immagine fittizia che rappresenta il numero 4 ---*/
    double *dummy_image = (double*)malloc(784 * sizeof(double));
    if (dummy_image == NULL) {
        fprintf(stderr, "Errore di allocazione per l'immagine perfetta.\n");
        exit(EXIT_FAILURE);
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

    double* dummy_4 = dummy_4_factory();
    double* dummy_4_shifted = dummy_4_shifted_factory();

    // Creazione Dato Target (One-Hot Encoding per la classe 4)
    double *target = (double*)malloc(10 * sizeof(double));
    if (target == NULL) exit(EXIT_FAILURE);
    for (int i = 0; i < 10; i++) target[i] = 0.0;
    target[4] = 1.0; // Vogliamo che il neurone con indice 4 si attivi

    printf("Inizio addestramento su singolo campione perfetto (1000 epoche)...\n");

    // ciclo di training
    int epochs = 1000;
    for (int e = 0; e < epochs; e++) {
        forward_pass(nn, dummy_4);
        backward_pass(nn, target);
        update_weights(nn);
    }

    // Test 1: Verifica sull'immagine perfetta (Controllo)
    printf("\n--- TEST 1: Forward pass su immagine originale ---\n");
    forward_pass(nn, dummy_4);
    Layer *output_layer = &nn->layers[num_layers - 1];
    printf("Probabilita' Neurone 4: %.4f\n", output_layer->neurons[4].output);

    // Test 2: Verifica sull'immagine sfasata (Test di Generalizzazione)
    printf("\n--- TEST 2: Forward pass su immagine sfasata di 1 pixel ---\n");
    forward_pass(nn, dummy_4_shifted);
    printf("Probabilita' Neurone 4 (sfasato): %.4f\n", output_layer->neurons[4].output);

    free(dummy_4);
    free(dummy_4_shifted);
    free(target);
    free_network(nn);
    printf("\nMemoria deallocata correttamente, 0 leaks.\n");

    return 0;
}