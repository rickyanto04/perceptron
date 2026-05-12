/*
Codice per singolo neurone addestrato a riconoscere l'AND logico da due input x1 e x2;
l'AND è un problema linearmente separabile, un solo neurone può risolverlo.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATASET_ROWS 4
#define DATASET_COLS 3


/*
struct per il singolo neurone:
    connessioni in ingresso
    array dinamico per i pesi sinaptici
    soglia di attivazione
    tasso di apprendimento (v. convergenza)
*/
typedef struct {
    int num_inputs;
    double *weights;
    double bias;
    double learning_rate;
} Neuron;

/*
funzione per la creazione di un neurone
*/
Neuron* create_neuron(const int inputs, const double lr) {

    //allocazione memoria per la struttura di base
    Neuron *n = (Neuron*)malloc(sizeof(Neuron));
    if (n == NULL) {
        fprintf(stderr, "error: unable to allocate memory for the neuron\n");
        exit(EXIT_FAILURE);
    }

    n->num_inputs = inputs;
    n->learning_rate = lr;

    //bias randomico
    n->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

    //allocazione memoria per l'array dei pesi
    n->weights = (double*)malloc(inputs * sizeof(double));
    if (n->weights == NULL) {
        fprintf(stderr, "error: unable to allocate the weights array\n");
        free(n);
        exit(EXIT_FAILURE);
    }

    //inizializzazione dei pesi in maniera causale tra -1.0 e 1.0
    for (int i = 0; i < inputs; i++) {
        n->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    return n;
}

/*
funzione di attivazione:
    permette di appiattire l'output tra 0 e 1 (lo rende una probabilità)
*/
double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

/*
funzione per il calcolo della derivata della sigmoide:
    necessaria per il training
*/
double sigmoid_derivative(const double output) {
    return output * (1.0 - output);
}

/*
funzione di inferenza:
    forward pass
*/
double forward(const Neuron *n, double *inputs) {
    double sum = n->bias;

    //combinazione lineare degli input coi pesi
    for (int i = 0; i < n->num_inputs; i++) {
        sum += inputs[i] * n->weights[i];
    }

    return sigmoid(sum);
}

/*
funzione per aggiornare i pesi del neurone in base all'errore:
    backpropagation
*/
void train(Neuron *n, double *inputs, const double target) {
    //calcolo l'output attuale
    double output = forward(n, inputs);

    //calcolo dell'errore
    double error = target - output;

    //calcolo del gradiente = errore * derivata dell'output
    double gradient = error * sigmoid_derivative(output);

    //aggiornamento pesi: nuovo_peso = vecchio_peso + (learning rate * gradiente * input)
    for (int i = 0; i < n->num_inputs; i++) {
        n->weights[i] += n->learning_rate * gradient * inputs[i];
    }

    //aggiornamento bias in modo simile a quello dei pesi
    n->bias += n->learning_rate * gradient;
}

/*
MSE - Mean Squared Error:
    calcola la media dei quadrati degli errori sull'intero dataset (precisione)
*/
double calculate_mse(const Neuron *n, const double dataset[DATASET_ROWS][DATASET_COLS], const int num_samples) {
    double total_error = 0.0;

    for (int i = 0; i < num_samples; i++) {
        double inputs[2] = {dataset[i][0], dataset[i][1]};
        double target = dataset[i][2];
        double prediction = forward(n, inputs); //forward pass senza aggiornamento dei pesi (fini statistici)

        total_error += pow(target - prediction, 2);
    }
    return total_error / num_samples;
}

void free_neuron(Neuron *n) {
    if (n != NULL) {
        free(n->weights);
        free(n);
    }
}

int main() {
    srand((unsigned int)time(NULL));

    //dataset: {input1, input2, target}
    const double dataset[DATASET_ROWS][DATASET_COLS] = {
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 1.0}
    };

    Neuron *n = create_neuron(2, 0.5);

    printf("--- TRAINING STARTED ---\n\n");
    printf("initial weights: w1 = %.4f | w2 = %.4f | bias = %.4f\n\n", n->weights[0], n->weights[1], n->bias);

    //ciclo di addestramento (10000 epoche)
    int epochs = 10000;
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < DATASET_ROWS; i++) {
            double inputs[2] = {dataset[i][0], dataset[i][1]};
            train(n, inputs, dataset[i][2]);
        }

        double current_mse = calculate_mse(n, dataset, DATASET_ROWS);

        //telemetria = stampiamo log ogni 1000 epoche
        if (e == 1 || e % 1000 == 0) {
            printf("epoch %5d | MSE: %.6f | w1: %7.4f | w2: %7.4f | bias: %7.4f\n",
            e, current_mse, n->weights[0], n->weights[1], n->bias);
        }

        if (current_mse < 0.001) {
            printf("early stopping activated at epoch %d (MSE: %.6f)\n", e, current_mse);
            break;
        }
    }

    //fase di test: inferenza dopo l'addestramento
    printf("\n--- TRAINING ENDED --- (logic AND) -> results:\n\n");
    for (int i = 0; i < DATASET_ROWS; i++) {
        double inputs[2] = {dataset[i][0], dataset[i][1]};
        double prediction = forward(n, inputs);
        printf("input: (%.1f, %.1f) -> prediction: %.4f (target: %.1f)\n",
            inputs[0], inputs[1], prediction, dataset[i][2]);
    }

    free_neuron(n);

    return 0;
}