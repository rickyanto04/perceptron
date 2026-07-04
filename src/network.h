#ifndef NETWORK_H
#define NETWORK_H

// Struttura per un singolo neurone
typedef struct {
    int num_inputs;
    double *weights;
    double bias;
    double output;     // Memorizza l'output durante la forward pass (fondamentale per la backprop)
    double error_grad; // Memorizza il gradiente dell'errore (delta) per questo neurone
} Neuron;

// Struttura per uno strato (Layer) di neuroni
typedef struct {
    int num_neurons;
    Neuron *neurons;
} Layer;

// Struttura per l'intera Rete Neurale
typedef struct {
    int num_layers;
    Layer *layers;
    double learning_rate;
} Network;

// creazione dell'intera rete
Network *create_network(int num_layers, int *layer_sizes, double lr);

// deallocazione sicura di tutta la memoria
void free_network(Network *net);

#endif