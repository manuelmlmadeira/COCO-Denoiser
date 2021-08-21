from numpy.linalg import norm
import numpy as np
import numpy.linalg as la
from time import time

class monitor:
    def __init__(self, algo, loss, x_min, args=()):
        self.x_min = x_min
        self.algo = algo
        self.loss = loss
        self.args = args
        self.f_min = loss(x_min, *args)
    
    def run(self, *algo_args, **algo_kwargs):
        t0 = time()
        _, x_list = self.algo(*algo_args, **algo_kwargs)
        self.total_time = time() - t0
        self.x_list = x_list
        self.err = [norm(x - self.x_min) for x in x_list]
        self.obj = [self.loss(x, *self.args) - self.f_min for x in x_list]


def grad_i_linreg(i, x, A, b, lbda):
    """Gradient with respect to a sample"""
    a_i = A[i]
    return (a_i.dot(x) - b[i]) * a_i + lbda * x


def grad_linreg(x, A, b, lbda):
    """Full gradient"""
    n = b.size
    g = np.zeros_like(x)
    for i in range(n):
        g += grad_i_linreg(i, x, A, b, lbda)
    return g / n


def loss_linreg(x, A, b, lbda):
    n = b.size
    return norm(A.dot(x) - b) ** 2 / (2. * n) + lbda * norm(x) ** 2 / 2.


def lipschitz_linreg(A, b, lbda):
    n = b.size
    return norm(A, ord=2) ** 2 / n + lbda


def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))


def grad_i_logreg(i, x, A, b, lbda):
    """Gradient with respect to a sample"""
    a_i = A[i]
    b_i = b[i]
    return - a_i * b_i / (1. + np.exp(b_i * np.dot(a_i, x))) + lbda * x



def grad_logreg(x, A, b, lbda):
    """Full gradient"""
    n = b.size
    u = b * A.dot(x)
    return A.T.dot(b * (sigmoid(u) - 1)) / n +  lbda * x

def loss_logreg(x, A, b, lbda):
    bAx = b * np.dot(A, x)
    return np.mean(np.log(1. + np.exp(- bAx))) + lbda * norm(x) ** 2 / 2.

def lipschitz_logreg(A, b, lbda):
    n = b.size
    return norm(A, ord=2) ** 2 / (4. * n) + lbda

def mu_constant_logreg(A, b, lbda):
    """Return the strong convexity constant"""
    n = b.size
    mu =  min(abs(la.eigvals(np.dot(A,A.T)))) / (4*n) + lbda
    return mu 

