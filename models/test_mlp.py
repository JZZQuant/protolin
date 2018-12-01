import time
from unittest import TestCase

from models.activations import sigmoid, square_loss
from models.datasets import toric_knots, load_cifar_10_data
from models.multilayerperceptron import MultilayerPerceptron
import numpy as np
import os


class TestMlp(TestCase):
    def test_forward_backward_prop(self):
        full, X, y = toric_knots(80000)
        split = int(X.shape[0] * 0.7)
        model = MultilayerPerceptron(X[:split], y[:split], [20, 20, 12], [sigmoid(), sigmoid(), sigmoid(), sigmoid()],
                                     square_loss())
        start_time = time.time()
        train_accuracy = model.train()
        test_accuracy = model.test(X[split:], y[split:])
        print("--- %s seconds took for training ---" % (time.time() - start_time))
        print("accuracy on test set: %.2f" % test_accuracy)
        assert (np.abs(test_accuracy - train_accuracy) < 5)

    def test_load_cifar10(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../data_dir/cifar-10-batches-py")
        cifar_train_data, cifar_train_filenames, cifar_train_labels, cifar_test_data, \
        cifar_test_filenames, cifar_test_labels, cifar_label_names = load_cifar_10_data(filename)
        print(cifar_label_names)
