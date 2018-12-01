import numpy as np


class sigmoid(object):

    def forward(self, x, weigths, bias):
        theta = np.dot(x, weigths) + bias
        return 1 / (1 + np.exp(-theta))

    def backward(self, signal):
        return signal * (1 - signal)


class relu(object):

    def forward(self, x, weigths, bias):
        theta = np.dot(x, weigths) + bias
        return np.maximum(theta, 0)

    def backward(self, signal):
        signal[signal > 0] = 1
        signal[signal <= 0] = 0.005
        return signal


class square_loss(object):
    def forward(self, signal, y):
        predict = [int(t > 0.5) for t in signal]
        incorrects = np.sum(np.abs(np.array(predict) - y.T))
        return (len(predict) - incorrects) * 100 / len(predict)

    def backward(self, signal, y):
        return (signal - y) / y.shape[0]
