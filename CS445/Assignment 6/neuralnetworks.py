import numpy as np

class NeuralNetwork:
    def __init__(self, n_inputs, hidden_layers, n_outputs):
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        np.random.seed(42)  # For reproducibility
        layers = [self.n_inputs] + self.hidden_layers + [self.n_outputs]
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i] + 1, layers[i + 1]) / np.sqrt(layers[i])  # +1 for bias
            weights.append(w)
        return weights

    def forward(self, inputs):
        activations = [np.hstack([inputs, np.ones((inputs.shape[0], 1))])]  # Add bias term to inputs
        for w in self.weights[:-1]:
            z = np.dot(activations[-1], w)
            a = np.tanh(z)  # Tanh activation function
            activations.append(np.hstack([a, np.ones((a.shape[0], 1))]))  # Add bias term to each hidden layer
        z = np.dot(activations[-1], self.weights[-1])
        activations.append(z)  # Output layer without non-linear activation
        return activations

    def backward(self, activations, targets):
        outputs = activations[-1]
        errors = outputs - targets
        delta = errors  # For linear output layer

        n_layers = len(self.weights)
        deltas = [delta]
        
        # Backpropagation
        for i in range(n_layers - 1, 0, -1):
            delta = deltas[-1].dot(self.weights[i].T)[:, :-1] * (1 - activations[i][:, :-1]**2)
            deltas.append(delta)

        deltas.reverse()
        gradients = []
        for i in range(n_layers):
            gradient = activations[i].T.dot(deltas[i]) / len(targets)
            gradients.append(gradient)
        
        return gradients

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i]

    def train(self, inputs, targets, n_epochs, learning_rate, method='sgd', verbose=False):
        for epoch in range(n_epochs):
            activations = self.forward(inputs)
            gradients = self.backward(activations, targets)
            self.update_weights(gradients, learning_rate)
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {np.mean((targets - activations[-1])**2)}")

    def use(self, inputs):
        return self.forward(inputs)[-1]  # Only return the output from the last layer

    def __repr__(self):
        return f'NeuralNetwork(n_inputs={self.n_inputs}, hidden_layers={self.hidden_layers}, n_outputs={self.n_outputs})'
