#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "neural_math.h"

Network *create_network(int num_layers, int *layer_sizes, double lr) {
    // allocazione struttura principale
    Network *net = (Network*)malloc(sizeof(Network));
    if (!net) { fprintf(stderr, "Errore allocazione Network\n"); exit(EXIT_FAILURE); }

    net->num_layers = num_layers;
    net->learning_rate = lr;

    // allocazione array Layers
    net->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    if (!net->layers) { fprintf(stderr, "Errore allocazione Layers\n"); exit(EXIT_FAILURE); }

    // costruzione di ogni singolo Layer
    for(int l = 0; l < num_layers; l++) {

        net->layers[l].num_neurons = layer_sizes[l];
        net->layers[l].neurons = (Neuron*)malloc(layer_sizes[l] * sizeof(Neuron));
        if (!net->layers[l].neurons) { fprintf(stderr, "Errore allocazione Neuroni\n"); exit(EXIT_FAILURE); }

        // creazione di ogni singolo neurone nel Layer corrente
        for(int n = 0; n < layer_sizes[l]; n++) {

            Neuron *neuron = &net->layers[l].neurons[n];
            neuron->output = 0.0;
            neuron->error_grad = 0.0;

            if (l == 0) {
                neuron->num_inputs = 0;
                neuron->weights = NULL;
                neuron->bias = 0.0;
            } else {

                neuron->num_inputs = layer_sizes[l - 1];
                neuron->weights = (double*)malloc(neuron->num_inputs * sizeof(double));
                if (!neuron->weights) { fprintf(stderr, "Errore allocazione Pesi\n"); exit(EXIT_FAILURE); }

                // inizializzazione casuale
                for (int w = 0; w < neuron->num_inputs; w++) {
                    neuron->weights[w] = ((double)rand() / RAND_MAX) - 0.5;
                }
                neuron->bias = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }

    return net;
}

void free_network(Network *net) {
    if (!net) return;

    // Deallocazione dal livello più profondo verso l'alto per evitare memory leak
    for (int l = 0; l < net->num_layers; l++) {
        for (int n = 0; n < net->layers[l].num_neurons; n++) {
            // Libero l'array dei pesi solo se esiste (saltando quindi il primo layer)
            if (net->layers[l].neurons[n].weights != NULL) {
                free(net->layers[l].neurons[n].weights);
            }
        }
        // Libero l'array dei neuroni per questo layer
        free(net->layers[l].neurons);
    }
    // Libero l'array dei layers
    free(net->layers);
    // Libero la struttura principale
    free(net);
}

void forward_pass(Network *net, double *input_data) {

    if (!net || !input_data) return;

    /* Popola il livello di Input (Layer 0)
       l'input non subisce elaborazioni, viene solo copiato nella rete*/
    Layer *input_layer = &net->layers[0];
    for (int i = 0; i < input_layer->num_neurons; i++) {
        input_layer->neurons[i].output = input_data[i];
    }

    // Propagazione in avanti per i layer successivi (Hidden e Output)
    for (int l = 1; l < net->num_layers; l++) {
        Layer *current_layer = &net->layers[l];
        Layer *prev_layer = &net->layers[l - 1]; // Il layer precedente fa da input

        for (int n = 0; n < current_layer->num_neurons; n++) {
            Neuron *neuron = &current_layer->neurons[n];
            double sum = neuron->bias;

            // Somma pesata: (peso * output_del_neurone_precedente)
            for (int w = 0; w < neuron->num_inputs; w++) {
                sum += neuron->weights[w] * prev_layer->neurons[w].output;
            }

            // Attivazione non lineare
            neuron->output = sigmoid(sum);
        }
    }
}