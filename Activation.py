import numpy as np
from Tensor import Tensor

class Activation_Relu:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

class Activation_Softmax:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
class ReLU:
    def __call__(self, x):
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

        def _backward():
            x.grad += (x.data > 0) * out.grad

        out._backward = _backward
        out._parents = [x]
        return out

    def parameters(self):
        return []

class Sigmoid:
    def __call__(self, x):
        sig = 1 / (1 + np.exp(-x.data))
        out = Tensor(sig, requires_grad=x.requires_grad)

        def _backward():
            x.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        out._parents = [x]
        return out

    def parameters(self):
        return []
