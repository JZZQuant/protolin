import numpy as np


class layer(object):
    def __init__(self,size,activation,learning_rate,momentum):
        self.weigths = np.random.normal(0,1,size)
        self.bias=np.random.normal(0,1,size[1])
        self.activation=activation
        self.learning_rate=learning_rate
        self.momentum = momentum
        self.weight_momentum=0
        self.bias_momentum=0

    def update(self,weight_update,bias_update ):
        self.weigths -= self.learning_rate * ( weight_update  + self.weight_momentum *self.momentum)
        self.bias -= self.learning_rate * (bias_update + self.bias_momentum * self.momentum)
        self.weight_momentum=weight_update
        self.bias_momentum=bias_update

    def forward(self,signal):
        return self.activation.forward(signal, self.weigths, self.bias)

    def backward(self,signal):
        return self.activation.backward(signal)