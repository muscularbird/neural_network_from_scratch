import cupy as cp
from Tensor import Tensor

class Activation_Relu:
    @staticmethod
    def forward(x: cp.ndarray) -> cp.ndarray:
        return cp.maximum(0, x)

class Activation_Softmax:
    @staticmethod
    def forward(x: cp.ndarray) -> cp.ndarray:
        exp_values = cp.exp(x - cp.max(x, axis=1, keepdims=True))
        probabilities = exp_values / cp.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
class ReLU:
    def __call__(self, x):
        out = Tensor(cp.maximum(0, x.data), requires_grad=x.requires_grad)

        def _backward():
            x.grad += (x.data > 0) * out.grad

        out._backward = _backward
        out._parents = [x]
        return out

    def parameters(self):
        return []

class Sigmoid:
    def __call__(self, x):
        sig = 1 / (1 + cp.exp(-x.data))
        out = Tensor(sig, requires_grad=x.requires_grad)

        def _backward():
            x.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        out._parents = [x]
        return out

    def parameters(self):
        return []
