#include <math.h>
#include "neural_math.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double output) {
    return output * (1.0 - output);
}