"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        if len(self.u) == 0:
            self.u[0] = np.zeros(len(self.params), dtype=np.float32)
        
        t = len(self.u) - 1
        self.u[t + 1] = []
        for i, w in enumerate(self.params):
            grad = w.grad + self.weight_decay * w.data
            
            self.u[t + 1].append(self.momentum * self.u[t][i] + np.float32(1 - self.momentum) * grad)
            self.u[t + 1][i] = self.u[t + 1][i].data
            
            w.data = w.data - self.lr * self.u[t + 1][i].data

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        

    def step(self):
        if len(self.m) == 0:
            self.m[0] = np.zeros(len(self.params), dtype=np.float32)
            
        if len(self.v) == 0:
            self.v[0] = np.zeros(len(self.params), dtype=np.float32)
        
        t = self.t
        self.m[t + 1] = []
        self.v[t + 1] = []
        
        for i, w in enumerate(self.params):
            grad = w.grad + self.weight_decay * w.data
            self.m[t + 1].append(self.beta1 * self.m[t][i] + np.float32(1 - self.beta1) * grad)
            self.m[t + 1][i] = self.m[t + 1][i].data
            
            self.v[t + 1].append(self.beta2 * self.v[t][i] + np.float32(1 - self.beta2) * grad**2)
            self.v[t + 1][i] = self.v[t + 1][i].data
            
            u_hat = self.m[t + 1][i] / np.float32(1 - self.beta1**(t + 1))
            v_hat = self.v[t + 1][i] / np.float32(1 - self.beta2**(t + 1))
            u_hat.detach()
            v_hat.detach()
            
            w.data = w.data - self.lr * u_hat / (v_hat**0.5 + self.eps)
            
        self.t += 1
