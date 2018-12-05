from structures.layer import layer
from utils.sampling import generate_batches
import numpy as np

np.random.seed(122)


class MultilayerPerceptron(object):
    def __init__(self, X, y, hidden_layers, activations, cost_function, learning_rate=0.1, momentum=0.9):
        dim = X.shape[1]
        classes = y.shape[1]
        complete_layers = [dim] + hidden_layers + [classes]
        weight_sizes = list(zip(complete_layers[:-1], complete_layers[1:]))
        self.layers = [layer(*l, learning_rate, momentum) for l in list(zip(weight_sizes, activations))]
        self.cost = layer((1, 1), cost_function, learning_rate, momentum)
        self.X = X
        self.y = y

    # Takes the signal in the current layer X and all the further layers/weights that are dependent on this layer
    # returns the gradients of each node to this layer and all the updated layers
    # refer to :http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
    def forward_backward_prop(self, features, y, layers):
        if len(layers) == 0:
            # For top most layer just return the error gradient on cost function
            return self.cost.activation.backward(features, y), []
        # pop the layer from stack
        current_layer = layers.pop(0)
        # Get the activated signals/features for next layers
        activated = current_layer.forward(features)

        # Get the gradients upto the activated signals of next layer through recursion
        fw_activated_gradient, updated_layers = self.forward_backward_prop(activated, y, layers)
        # Multiply with the the "gradient of the activation" = gradients upto unactivated signals of the next layers
        fw_unactivated_gradient = fw_activated_gradient * current_layer.backward(activated)

        # Pass the gradients of previous layer upto the weights, by a row wise kron product to each vector in the batch.
        # kronecker product of size_m gradients of next layer and size_n signal of the this layers gives a mXn weights.
        # Weights are accumulated over all data point in the batch using an einstein summation.
        weight_updates = np.einsum('bi,bo->io', features, fw_unactivated_gradient, optimize=True)

        # Compute gradients upto the current layers activated signals before updating the weights
        gradients = np.dot(fw_activated_gradient, current_layer.weigths.T)

        # Update weights and bias terms
        current_layer.update(weight_updates, np.sum(fw_unactivated_gradient, axis=0))
        # Recursively send the updated layers back to the previous layer
        updated_layers.append(current_layer)
        return gradients, updated_layers

    def train(self):
        epoch = 0
        accuracy = -1
        while accuracy < 95:
            epoch += 1
            for batch in generate_batches(self.X, 100):
                grads, layers = self.forward_backward_prop(self.X[batch], self.y[batch], self.layers)
                self.layers = layers[::-1]
            accuracy = self.test(self.X, self.y)
            if epoch % 10 == 0: print("-------- epoch : %d , accuracy : %.8f --------- " % (epoch, accuracy))
        print("-------- epoch : %d , accuracy : %.8f --------- " % (epoch, accuracy))
        return accuracy

    def test(self, X, y):
        for layer in self.layers:
            X = layer.activation.forward(X, layer.weigths, layer.bias)
        return self.cost.activation.forward(X, y)
