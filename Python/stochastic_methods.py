import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from scipy.optimize import rosen as f
from scipy.optimize import rosen_der as fp
from scipy.optimize import rosen_hess as fpp
from numpy import linalg as LA
from numpy import random
import time



## Baseline Algorithms ###############################

def str_sgd(n, x_init,  grad_i, store_every, step_size, n_steps,  args=()):
    """Stochastic gradient descent algorithm."""
    x = x_init.copy()
    x_list = [x_init]
    iis = np.random.randint(n, size=n_steps)
    for step_idx in range(n_steps):
        example_idx = iis[step_idx]
        grad = grad_i(example_idx, x, *args) 
        x += - step_size * grad
        
        if (step_idx+1)%store_every == 0:
            x_list.append(x.copy())
    
    return x, x_list

def str_adam(x_init, grad_i,  store_every, step, n_steps, beta1, beta2, epsilon,  args=()):
    """Streaming Adam"""
    x = x_init.copy()
    d = x.size
    m = np.zeros(d)
    v = np.zeros(d)
    x_list = [x_init]
    iis = np.arange(n_steps)
    random.shuffle(iis)
    for step_idx in range(n_steps):
        example_idx = iis[step_idx]
        g = grad_i(example_idx, x, *args) 
        
        iteration = step_idx + 1
        m = beta1*m +(1-beta1)*g 
        v = beta2*v +(1-beta2)*(g*g)
        mhat= m/(1-beta1**(iteration))
        vhat= v/(1-beta2**(iteration))
        direction = -step*mhat/(np.sqrt(vhat) +epsilon)
        x += direction
        if (step_idx+1)%store_every == 0:
            x_list.append(x.copy())
    
    return x, x_list


def str_saga(rho, n, x_init,  grad_i, store_every, step_size, n_steps,  args=()):
    """SAGA"""
    x = x_init.copy()
    x_list = [x_init]
    d = x.size
    gradient_memory = np.zeros((n, d))
    averaged_gradient = np.zeros(d)
    iis = np.arange(n)
    random.shuffle(iis)
    sample_set_idxs = []
    buffer_idxs = []
    for time_step in range(int(n_steps/rho)):
        new_example_idx = random.choice(iis)
        buffer_idxs.append(new_example_idx)
        iis = np.delete(iis, np.where(iis == new_example_idx))
        
        for j in range(rho):
            
            if j % 2 == 0 and len(buffer_idxs) > 0:
                sample_set_idxs.append(new_example_idx)
                buffer_idxs.remove(new_example_idx)
                #gradient of the new index is initialized to 0
                averaged_gradient = (1-1/len(sample_set_idxs)) * averaged_gradient
                
            example_idx = random.choice(sample_set_idxs)
            grad_i_new = grad_i(example_idx, x, *args) 
            x += - step_size * (grad_i_new - gradient_memory[example_idx] + averaged_gradient)
            gradient_memory[example_idx] = grad_i_new

            x_list.append(x.copy())  
        
    return x, x_list


## COCO + Baseline Algorithms ###############################

def str_sgd_COCO(n, K, L, tol, x_init,  grad_i, store_every, step_size, n_steps,  args=()):
    """Stochastic gradient descent algorithm w/ COCO"""
    d = np.size(x_init)
    x = x_init.copy()
    x_list = [x_init]
    iis = np.random.randint(n, size=n_steps)
    G = np.zeros((d,K))
    X = np.zeros((d,K))
    
    # Building definitive structured matrix A and its L
    number_of_constraints = int(K * (K-1)/2)
    A = np.zeros((number_of_constraints * d, K * d))
    for k in range(K):
        for l in range(k+1,K):
            nb = int(- k**2/2 + k*(K-1/2) + l-(k+1))
            A[nb*d:(nb+1)*d, k*d:(k+1)*d ] = np.identity(d)
            A[nb*d:(nb+1)*d, l*d:(l+1)*d ] = -np.identity(d)
    L_A = LA.norm(A, 2)**2
    prev_dual = np.zeros((number_of_constraints * d, 1)) 
    
    for step_idx in range(n_steps):
        
        example_idx = iis[step_idx]
        new_grad = grad_i(example_idx, x, *args) 
        
        # Denoise
        G = np.roll(G, -1, axis=1)
        X = np.roll(X, -1, axis=1)
        G[:, K-1] = new_grad
        X[:, K-1] = x
        i_first = max([0, K-1-step_idx])
        considered_gradients = G[:,i_first:]
        considered_points = X[:,i_first:]
        n_considered_points = K - i_first
        if n_considered_points != K or n_considered_points == 2:
            denoised_gradients = COCOdenoiser(n_considered_points, d, L, tol, considered_gradients, considered_points)
        else:
            denoised_gradients, prev_dual = COCOdenoiser(n_considered_points, d, L, tol, considered_gradients, considered_points, A = A, L_A = L_A, prev_dual = prev_dual)
        denoised_grad = denoised_gradients[:, n_considered_points-1]
        # End Denoise
        
        x += - step_size * denoised_grad
        if (step_idx+1)%store_every == 0:
            x_list.append(x.copy())
    
    return x, x_list

def str_adam_COCO(n, K, L, tol, x_init, grad_i,  store_every, step, n_steps, beta1, beta2, epsilon,  args=()):
    """Streaming Adam"""
    x = x_init.copy()
    d = x.size
    m = np.zeros(d)
    v = np.zeros(d)
    x_list = [x_init]
    iis = np.random.randint(n, size=n_steps)
    G = np.zeros((d,K))
    X = np.zeros((d,K))
    
    # Building definitive structured matrix A and its L
    number_of_constraints = int(K * (K-1)/2)
    A = np.zeros((number_of_constraints * d, K * d))
    for k in range(K):
        for l in range(k+1,K):
            nb = int(- k**2/2 + k*(K-1/2) + l-(k+1))
            A[nb*d:(nb+1)*d, k*d:(k+1)*d ] = np.identity(d)
            A[nb*d:(nb+1)*d, l*d:(l+1)*d ] = -np.identity(d)
    L_A = LA.norm(A, 2)**2
    prev_dual = np.zeros((number_of_constraints * d, 1)) 
    
    for step_idx in range(n_steps):
        example_idx = iis[step_idx]
        new_g = grad_i(example_idx, x, *args)
        
        # Denoise
        G = np.roll(G, -1, axis=1)
        X = np.roll(X, -1, axis=1)
        G[:, K-1] = new_g
        X[:, K-1] = x
        i_first = max([0, K-1-step_idx])
        considered_gradients = G[:,i_first:]
        considered_points = X[:,i_first:]
        n_considered_points = K - i_first
        if n_considered_points != K or n_considered_points == 2:
            denoised_gradients = COCOdenoiser(n_considered_points, d, L, tol, considered_gradients, considered_points)
        else:
            denoised_gradients, prev_dual = COCOdenoiser(n_considered_points, d, L, tol, considered_gradients, considered_points, A = A, L_A = L_A, prev_dual = prev_dual)
        g = denoised_gradients[:, n_considered_points-1]
        # End Denoise
        
        # Adam update
        iteration = step_idx + 1
        m = beta1*m +(1-beta1)*g 
        v = beta2*v +(1-beta2)*(g*g)
        mhat= m/(1-beta1**(iteration))
        vhat= v/(1-beta2**(iteration))
        direction = -step*mhat/(np.sqrt(vhat) +epsilon)
        x += direction
        if (step_idx+1)%store_every == 0:
            x_list.append(x.copy())
    
    return x, x_list


def str_saga_COCO(K, L, tol, rho, n, x_init,  grad_i, store_every, step_size, n_steps,  args=()):
    """SAGA"""
    x = x_init.copy()
    x_list = [x_init]
    d = x.size
    gradient_memory = np.zeros((n, d))
    averaged_gradient = np.zeros(d)
    iis = np.random.randint(n, size=n_steps)
    sample_set_idxs = []
    buffer_idxs = []
    G = np.zeros((d,K))
    X = np.zeros((d,K))
    
    # Building definitive structured matrix A and its L
    number_of_constraints = int(K * (K-1)/2)
    A = np.zeros((number_of_constraints * d, K * d))
    for k in range(K):
        for l in range(k+1,K):
            nb = int(- k**2/2 + k*(K-1/2) + l-(k+1))
            A[nb*d:(nb+1)*d, k*d:(k+1)*d ] = np.identity(d)
            A[nb*d:(nb+1)*d, l*d:(l+1)*d ] = -np.identity(d)
    L_A = LA.norm(A, 2)**2
    prev_dual = np.zeros((number_of_constraints * d, 1)) 
    
    
    for time_step in range(int(n_steps/rho)):
        new_example_idx = iis[time_step]
        buffer_idxs.append(new_example_idx)
        
        for j in range(rho):
            
            if j % 2 == 0 and len(buffer_idxs) > 0:
                sample_set_idxs.append(new_example_idx)
                buffer_idxs.remove(new_example_idx)
                #gradient of the new index is initialized to 0
                averaged_gradient = (1-1/len(sample_set_idxs)) * averaged_gradient
                
            example_idx = random.choice(sample_set_idxs)
            grad_i_new = grad_i(example_idx, x, *args)   
            
            # Denoise
            G = np.roll(G, -1, axis=1)
            X = np.roll(X, -1, axis=1)
            G[:, K-1] = grad_i_new
            X[:, K-1] = x
            i_first = max([0, K-1-time_step])
            considered_gradients = G[:,i_first:]
            considered_points = X[:,i_first:]
            n_considered_points = K - i_first
            if n_considered_points != K or n_considered_points == 2:
                denoised_gradients = COCOdenoiser(n_considered_points, d, L, tol, considered_gradients, considered_points)
            else:
                denoised_gradients, prev_dual = COCOdenoiser(n_considered_points, d, L, tol, considered_gradients, considered_points, A = A, L_A = L_A, prev_dual = prev_dual)
            denoised_grad = denoised_gradients[:, n_considered_points-1]
            # End Denoise
            
            x += - step_size * (denoised_grad - gradient_memory[example_idx] + averaged_gradient)
            gradient_memory[example_idx] = denoised_grad

            if (time_step+1)%store_every == 0:
                x_list.append(x.copy())
        
    return x, x_list



## COCO Implementation ####################

def COCOdenoiser( K, d, L, tol, G, X, A = 0, L_A = 0, prev_dual = 0):
    if K == 1:
        return G
    
    elif K == 2:
        g_1 = G[:,0]
        g_2 = G[:,1]
        x_1 = X[:,0]
        x_2 = X[:,1]
        r = L * np.dot(g_1-g_2, x_1 - x_2)
        if LA.norm(g_1 - g_2)**2 <= r:
            return G
        else:
            theta_1 = (g_1 + g_2 + L/2 * (x_1 - x_2))/2 + 0.5 *LA.norm(L/2 * (x_1-x_2)) * (g_1-g_2-L/2 *(x_1-x_2))/LA.norm((g_1-g_2-L/2 *(x_1-x_2)))
            theta_2 = (g_1 + g_2 - L/2 * (x_1 - x_2))/2 - 0.5 *LA.norm(L/2 * (x_1-x_2)) * (g_1-g_2-L/2 *(x_1-x_2))/LA.norm((g_1-g_2-L/2 *(x_1-x_2)))
            return np.array([theta_1, theta_2]).T
    
    else:
        number_of_constraints = int(K * (K-1)/2)
        ws_bool = L_A != 0
        
        s0 = np.zeros((number_of_constraints * d, 1))
        if ws_bool:
            prev_idx = (K-1)*d
            prev_len = (K-2)*d
            new_idx = 0
            while prev_idx < number_of_constraints*d :
                s0[new_idx:new_idx+prev_len] = prev_dual[prev_idx:prev_idx+prev_len]
                prev_idx = prev_idx+prev_len
                new_idx = new_idx+(prev_len+d)
                prev_len -= d
        
        # Build A, c, and r and compute L_A
        if not ws_bool:
            A = np.zeros((number_of_constraints * d, K * d))
        c = np.zeros((number_of_constraints * d, 1))
        r = np.zeros((number_of_constraints, 1))
        for k in range(K):
            for l in range(k+1,K):
                nb = int(- k**2/2 + k*(K-1/2) + l-(k+1))
                if L_A == 0:
                    A[nb*d:(nb+1)*d, k*d:(k+1)*d ] = np.identity(d)
                    A[nb*d:(nb+1)*d, l*d:(l+1)*d ] = -np.identity(d)
                c[nb*d:(nb+1)*d, 0] = (G[:,k] - L/2 * X[:,k]) - (G[:,l]- L/2 * X[:,l])
                r[nb , 0] = L/2 * LA.norm(X[:,k]-X[:,l])
        if not ws_bool:
            L_A = LA.norm(A, 2)**2
        
        
        
        
        #Solve Dual Problem with Fista
        s_star = FDPG(K, d, A, c, r, s0, L_A, tol)
        alpha_star = - np.dot(A.T, s_star)
        
        # Obtain Primal Solution
        ALPHA = np.reshape(alpha_star, (d, K), 'F')

        if not ws_bool:
            return ALPHA + G
        else:
            return ALPHA + G, s_star
    
def FDPG(K, d, A, c, r, s0, L_A, tol):
    s = s0
    y = s0
    t = 1
    delta_f = tol+1
    f_old = DualFunctionEval(K, d, A, c, r, s)
    counter =0
    while delta_f > tol:
        counter +=1
        p0 = y - 1/L_A * np.dot(A @ A.T, y)
        p = c + L_A*p0
        for k in range(K):
            for l in range(k+1,K):
                nb = int(- k**2/2 + k*(K-1/2) + l-(k+1))
                pkl1 = nb*d
                pkl2 = (nb+1)*d
                pkl = p[pkl1:pkl2, 0]
                rkl = r[nb, 0]
                if LA.norm(pkl) > rkl:
                    p[pkl1:pkl2, 0] = rkl * pkl/LA.norm(pkl)
                
        s_new = (c/L_A) + p0 -p/L_A
        
        # Update t
        t_new = (1 + (1+4* (t**2))**0.5)/2
        
        # Update y
        y_new = s_new + ((t-1)/t_new) * (s_new -s)
        
        # Prepare for next iteration
        s = s_new
        t = t_new
        
        f_new = DualFunctionEval(K, d, A, c, r, s)
        delta_f = LA.norm(f_new - f_old)
        f_old = f_new
        
    return s

def DualFunctionEval(K, d, A, c, r, s):
    p_conj = 0.5 * np.dot(s.T, np.dot(A @ A.T, s))
    q_conj = 0
    for k in range(K):
        for l in range(k+1,K):
            nb = int(- k**2/2 + k*(K-1/2) + l-(k+1))
            skl1 = nb*d
            skl2 = (nb+1)*d
            skl = s[skl1:skl2,0]
    
            ckl1 = nb*d
            ckl2 = (nb+1)*d 
            ckl = c[ckl1:ckl2,0]
            
            rkl = r[nb, 0]
            
            q_conj = q_conj + rkl*LA.norm(skl) - np.dot(ckl,skl)
            
    return p_conj + q_conj


## Data generation ###############################

from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz

def simu_linreg(x, n, std=1., corr=0.5):

    """Simulation for the least-squares problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix
    
    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)
    noise = std * randn(n)
    b = A.dot(x) + noise
    return A, b



def simu_logreg(x, n, std=1., corr=0.5):
    """Simulation for the logistic regression problem.
    
    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size    
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix
    
    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
        
        
    b : ndarray, shape (n,)
        The targets.
    """    
    A, b = simu_linreg(x, n, std=1., corr=corr)
    return A, np.sign(b)

# Get OpenML data

from sklearn.datasets import fetch_openml
import pandas as pd
from scipy.io import arff

def get_OpenML_data(name):
    dataset = arff.loadarff(name)
    Adf = pd.DataFrame(dataset[0])
    Ay = Adf.as_matrix()
    n_rows, n_cols = Ay.shape
    A = Ay[:,:-1]
    A = np.float_(A)
    b = Ay[:,-1]
    b = (np.int_(b))*2-3
    return A, np.sign(b)

    

