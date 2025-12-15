from Tensor import Tensor
import cupy as cp

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(cp.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.b = Tensor(cp.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]