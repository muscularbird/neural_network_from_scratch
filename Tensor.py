import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._parents = []

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        out._parents = [self, other]
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        out._parents = [self, other]
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        out._backward = _backward
        out._parents = [self, other]
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            # use 2D intermediate to handle 1D/2D combinations
            A = np.atleast_2d(self.data)
            B = np.atleast_2d(other.data)
            G = np.atleast_2d(out.grad)

            grad_A = G @ B.T
            grad_B = A.T @ G

            if self.data.ndim == 1:
                grad_A = grad_A.reshape(self.data.shape)
            if other.data.ndim == 1:
                grad_B = grad_B.reshape(other.data.shape)

            if self.requires_grad:
                self.grad += grad_A
            if other.requires_grad:
                other.grad += grad_B

        out._backward = _backward
        out._parents = [self, other]
        return out

    def __rmatmul__(self, other):
        # support numpy-array @ Tensor or scalar @ Tensor by converting left operand
        left = other if isinstance(other, Tensor) else Tensor(other)
        return left.__matmul__(self)

    # backward pass
    def backward(self):
        self.grad = np.ones_like(self.data)
        stack = [self]
        visited = set()
        while stack:
            t = stack.pop()
            if t not in visited:
                t._backward()
                visited.add(t)
                stack.extend(t._parents)

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        out._parents = [self]
        return out