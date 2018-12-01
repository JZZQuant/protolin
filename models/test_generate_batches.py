from unittest import TestCase
import numpy as np

from models.sampling import generate_batches


class TestGenerate_batches(TestCase):
    def test_generate_batches(self):
        print(list(generate_batches(np.arange(10,60),20)))