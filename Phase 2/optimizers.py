import numpy as np
from typing import Callable, Optional

# --- HELPER: Numerical Gradient ---
def get_numerical_gradient(cost_func: Callable, params: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_p = params.copy(); p_p[i] += epsilon
        p_m = params.copy(); p_m[i] -= epsilon
        grad[i] = (cost_func(p_p) - cost_func(p_m)) / (2 * epsilon)
    return grad

# --- HELPER: Numerical Hessian ---
def get_numerical_hessian(cost_func: Callable, params: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    n = len(params)
    hess = np.zeros((n, n))
    for i in range(n):
        p = params.copy()
        p[i] += epsilon; g_p = get_numerical_gradient(cost_func, p, epsilon)
        p[i] -= 2*epsilon; g_m = get_numerical_gradient(cost_func, p, epsilon)
        hess[:, i] = (g_p - g_m) / (2 * epsilon)
    return hess

# --- BASE OPTIMIZER ---
class Optimizer:
    def __init__(self, max_iters=1000, tol=1e-6):
        self.max_iters = max_iters
        self.tol = tol
        self.history = []
    def minimize(self, cost_func, initial_params, gradient_function=None):
        raise NotImplementedError

# --- 1. ADAM ---
class Adam(Optimizer):
    def __init__(self, learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iters=1000, tol=1e-6):
        super().__init__(max_iters, tol)
        self.lr, self.b1, self.b2, self.eps = learning_rate, beta1, beta2, epsilon
    def minimize(self, cost_func, init_params, gradient_function=None):
        p = np.array(init_params)
        m, v = np.zeros_like(p), np.zeros_like(p)
        if gradient_function is None: gradient_function = lambda x: get_numerical_gradient(cost_func, x)
        for t in range(1, self.max_iters+1):
            c = cost_func(p)
            g = gradient_function(p)
            self.history.append({'params': p.tolist(), 'cost': c})
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * (g**2)
            m_h = m / (1 - self.b1**t)
            v_h = v / (1 - self.b2**t)
            p -= self.lr * m_h / (np.sqrt(v_h) + self.eps)
            if np.linalg.norm(g) < self.tol: break
        return p

# --- 2. BFGS ---
class BFGS(Optimizer):
    def minimize(self, cost_func, init_params, gradient_function=None):
        p = np.array(init_params)
        dim = len(p)
        H_inv = np.eye(dim)
        if gradient_function is None: gradient_function = lambda x: get_numerical_gradient(cost_func, x)
        g = gradient_function(p)
        for _ in range(self.max_iters):
            self.history.append({'params': p.tolist(), 'cost': cost_func(p)})
            if np.linalg.norm(g) < self.tol: break
            d = -H_inv @ g
            p_new = p + d # Step size 1.0
            g_new = gradient_function(p_new)
            s, y = p_new - p, g_new - g
            if np.dot(y, s) > 1e-10: # Stability check
                rho = 1.0 / np.dot(y, s)
                I = np.eye(dim)
                H_inv = (I - rho * np.outer(s, y)) @ H_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            p, g = p_new, g_new
        return p

# --- 3. NELDER-MEAD ---
class NelderMead(Optimizer):
    def minimize(self, cost_func, init_params, gradient_function=None):
        dim = len(init_params)
        simplex = [np.array(init_params)] + [np.array(init_params) + (np.eye(dim)[i] * 0.05) for i in range(dim)]
        for _ in range(self.max_iters):
            simplex.sort(key=lambda x: cost_func(x))
            self.history.append({'params': simplex[0].tolist(), 'cost': cost_func(simplex[0])})
            centroid = np.mean(simplex[:-1], axis=0)
            best, worst = simplex[0], simplex[-1]
            # Reflect
            xr = centroid + 1.0 * (centroid - worst)
            if cost_func(best) <= cost_func(xr) < cost_func(simplex[-2]):
                simplex[-1] = xr
            elif cost_func(xr) < cost_func(best): # Expand
                xe = centroid + 2.0 * (xr - centroid)
                simplex[-1] = xe if cost_func(xe) < cost_func(xr) else xr
            else: # Contract
                xc = centroid + 0.5 * (worst - centroid)
                if cost_func(xc) < cost_func(worst): simplex[-1] = xc
                else: # Shrink
                    simplex = [best + 0.5 * (x - best) for x in simplex]
        return simplex[0]

# --- 4. NEWTON METHOD ---
class NewtonMethod(Optimizer):
    def minimize(self, cost_func, init_params, gradient_function=None, hessian_function=None):
        p = np.array(init_params)
        if gradient_function is None: gradient_function = lambda x: get_numerical_gradient(cost_func, x)
        if hessian_function is None: hessian_function = lambda x: get_numerical_hessian(cost_func, x)
        for _ in range(self.max_iters):
            self.history.append({'params': p.tolist(), 'cost': cost_func(p)})
            g = gradient_function(p)
            if np.linalg.norm(g) < self.tol: break
            try: d = np.linalg.solve(hessian_function(p), -g)
            except: d = -g * 0.01 # Fallback
            p += d
        return p

# --- 5. SADDLE-FREE NEWTON ---
class SaddleFreeNewton(NewtonMethod):
    def minimize(self, cost_func, init_params, gradient_function=None, hessian_function=None):
        p = np.array(init_params)
        if gradient_function is None: gradient_function = lambda x: get_numerical_gradient(cost_func, x)
        if hessian_function is None: hessian_function = lambda x: get_numerical_hessian(cost_func, x)
        for _ in range(self.max_iters):
            self.history.append({'params': p.tolist(), 'cost': cost_func(p)})
            g = gradient_function(p)
            if np.linalg.norm(g) < self.tol: break
            H = hessian_function(p)
            evals, evecs = np.linalg.eigh(H)
            abs_evals = np.abs(evals)
            abs_evals[abs_evals < 1e-4] = 1e-4 # Regularize
            H_safe_inv = evecs @ np.diag(1.0/abs_evals) @ evecs.T
            p -= H_safe_inv @ g
        return p

# --- 6. SPSA (Research Version) ---
class SPSA(Optimizer):
    def __init__(self, a=0.6, c=0.1, alpha=0.602, gamma=0.101, max_iters=200):
        super().__init__(max_iters)
        self.a, self.c, self.alpha, self.gamma = a, c, alpha, gamma
    def minimize(self, cost_func, init_params, gradient_function=None):
        p = np.array(init_params)
        for k in range(self.max_iters):
            ak, ck = self.a / (k+1+100)**self.alpha, self.c / (k+1)**self.gamma
            delta = 2 * np.random.randint(0, 2, size=len(p)) - 1
            y_p, y_m = cost_func(p + ck*delta), cost_func(p - ck*delta)
            g = (y_p - y_m) / (2 * ck * delta)
            p -= ak * g
            self.history.append({'params': p.tolist(), 'cost': (y_p+y_m)/2})
        return p

# --- 7. HYBRID (Adam -> BFGS) ---
class HybridOptimizer(Optimizer):
    def __init__(self, switch_iter=50, max_iters=200):
        super().__init__(max_iters)
        self.switch = switch_iter
        self.adam = Adam(max_iters=switch_iter, learning_rate=0.05)
        self.bfgs = BFGS(max_iters=max_iters-switch_iter)
    def minimize(self, cost_func, init_params, gradient_function=None):
        p = self.adam.minimize(cost_func, init_params, gradient_function)
        self.history.extend(self.adam.history)
        p = self.bfgs.minimize(cost_func, p, gradient_function)
        self.history.extend(self.bfgs.history)
        return p