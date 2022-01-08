from logistic_linear import *
from stochastic_methods import str_sgd, str_sgd_COCO, str_adam, str_adam_COCO, str_saga, str_saga_COCO
from utils import get_data, find_precise_minimum, run_method

# Setup
np.random.seed(0)
dataset_name = "fourclass"
lbda = 0  # 1.00001 / n ** (0.5)
x_0 = 1
use_cvx = False

# Initialization
loss = loss_logreg
grad = grad_logreg
grad_i = grad_i_logreg
A, b = get_data(dataset_name)
n, d = A.shape
A = A.toarray()
mu = mu_constant_logreg(A, b, lbda)
L = lipschitz_logreg(A, b, lbda)
x_min, f_min = find_precise_minimum(A, b, lbda, loss, grad)
x_0 = x_0 * np.ones(d)
print("Size of the Dataset (n, d) = ", n, d)
print("\lambda = %s" % lbda)
print("\mu = %s" % mu)
print("L = %s" % L)

# SGD
method = str_sgd
method_plus_COCO = str_sgd_COCO
method_str = "SGD"
n_reps = 100  # 50
x_init = x_0.copy()  # x_min + x_0.copy() * 0.5 * 10 ** (-1)
store_every = 1
n_steps = 800  # 1000
K_list = [1, 2, 4, 8]
tol = 10 ** 0
# step_size
method_hyperparams = 1 * 1 / L  # 2 * 1 / L

run_method(dataset_name, method, method_plus_COCO, method_str, n_reps, x_init, store_every, n_steps, K_list, L, tol,
           method_hyperparams, n, loss, grad_i, A, b, lbda, x_min, f_min, use_cvx)

# Adam
method = str_adam
method_plus_COCO = str_adam_COCO
method_str = "Adam"
n_reps = 100  # 50
x_init = x_0.copy()
store_every = 1
n_steps = 800  # 200
K_list = [1, 2, 4, 8, 16]  # [1, 2, 4, 8]
tol = 10 ** 0
# step_size, beta1, beta2, epsilon
method_hyperparams = (80 * 1 / L, 0.9, 0.999, 10 ** (-8))  # (10 ** (-1), 0.9, 0.999, 10 ** (-8))

run_method(dataset_name, method, method_plus_COCO, method_str, n_reps, x_init, store_every, n_steps, K_list, L, tol,
           method_hyperparams, n, loss, grad_i, A, b, lbda, x_min, f_min, use_cvx)

# STRSAGA
method = str_saga
method_plus_COCO = str_saga_COCO
method_str = "STRSAGA"
n_reps = 100  # 50
x_init = x_0.copy()  # x_min + x_0.copy() * 0.5 * 10 ** (-1)
store_every = 1
n_steps = 800  # 1000
K_list = [1, 2, 4, 8]
tol = 10 ** 0
# step_size, rho
method_hyperparams = (1 * 1 / L, 1)  # (4 * 1 / L, 1)

run_method(dataset_name, method, method_plus_COCO, method_str, n_reps, x_init, store_every, n_steps, K_list, L, tol,
           method_hyperparams, n, loss, grad_i, A, b, lbda, x_min, f_min, use_cvx)
