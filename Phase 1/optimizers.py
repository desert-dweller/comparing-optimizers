import numpy as np
from typing import Callable, Optional

# --- HELPER 1: Numerical Gradient (Finite Difference) ---
def get_numerical_gradient(cost_func: Callable, params: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Calculates the gradient vector using the Central Finite Difference method.
    Formula: f'(x) ≈ (f(x + e) - f(x - e)) / 2e
    """
    grad = np.zeros_like(params)
    params_copy = params.copy()
    
    for i in range(len(params)):
        original_val = params_copy[i]
        
        # Step Forward & Backward
        params_copy[i] = original_val + epsilon
        cost_plus = cost_func(params_copy)
        params_copy[i] = original_val - epsilon
        cost_minus = cost_func(params_copy)
        
        # Slope calculation
        grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
        params_copy[i] = original_val # Reset
        
    return grad

# --- HELPER 2: Numerical Hessian (Finite Difference of Gradient) ---
def get_numerical_hessian(cost_func: Callable, params: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """
    Calculates the Hessian matrix by taking the finite difference of the gradient.
    Formula: H ≈ (∇f(x + e) - ∇f(x - e)) / 2e
    """
    n = len(params)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        params_copy = params.copy()
        
        # Perturb parameter i
        params_copy[i] += epsilon
        grad_plus = get_numerical_gradient(cost_func, params_copy, epsilon)
        
        params_copy[i] -= 2 * epsilon
        grad_minus = get_numerical_gradient(cost_func, params_copy, epsilon)
        
        params_copy[i] += epsilon # Reset
        
        # Fill column i
        hessian[:, i] = (grad_plus - grad_minus) / (2 * epsilon)
        
    return hessian

# --- OPTIMIZER CLASSES ---
class Optimizer:
    """
    Base class for all custom optimizers.
    Enforces a standard interface for benchmarking.
    """
    def __init__(self, max_iters: int = 1000, tol: float = 1e-6):
        self.max_iters = max_iters
        self.tol = tol
        self.history = [] 

    def minimize(self, cost_function: Callable, initial_params: np.ndarray, gradient_function: Optional[Callable] = None) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method.")

class NewtonMethod(Optimizer):
    """
    Second-Order Optimizer (Exact).
    Update Rule: x_{k+1} = x_k - H^{-1} * ∇f(x_k)
    """
    def minimize(self, cost_function, initial_params, gradient_function=None, hessian_function=None):
        params = np.array(initial_params)
        
        if gradient_function is None:
            gradient_function = lambda p: get_numerical_gradient(cost_function, p)
        if hessian_function is None:
            hessian_function = lambda p: get_numerical_hessian(cost_function, p)
        
        for i in range(self.max_iters):
            cost = cost_function(params)
            grad = gradient_function(params)
            hess = hessian_function(params)
            self.history.append({'params': params.copy().tolist(), 'cost': cost})
            
            # Check Gradient Convergence
            if np.linalg.norm(grad) < self.tol:
                break
                
            # Solve linear system H * step = g (equivalent to step = H^-1 * g)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Fallback if Hessian is singular
                step = grad * 0.01 
                
            params = params - step
            
        return params

class BFGS(Optimizer):
    """
    Quasi-Newton Optimizer.
    Approximates the Inverse Hessian (H^-1) using the rank-2 BFGS update formula.
    """
    def __init__(self, max_iters: int = 1000, tol: float = 1e-6):
        super().__init__(max_iters, tol)

    def minimize(self, cost_function, initial_params, gradient_function=None, hessian_function=None):
        params = np.array(initial_params)
        dim = len(params)
        H_inv = np.eye(dim) # Start with Identity matrix approximation
        
        if gradient_function is None:
            gradient_function = lambda p: get_numerical_gradient(cost_function, p)
        
        grad = gradient_function(params)
        
        for i in range(self.max_iters):
            cost = cost_function(params)
            self.history.append({'params': params.copy().tolist(), 'cost': cost})
            
            if np.linalg.norm(grad) < self.tol:
                break
            
            # 1. Search Direction
            p = -H_inv @ grad
            
            # 2. Step (Fixed Alpha = 1.0 for Textbook Implementation)
            alpha = 1.0 
            params_new = params + alpha * p
            grad_new = gradient_function(params_new)
            
            # 3. Update H_inv approximation
            s = params_new - params # Change in position
            y = grad_new - grad     # Change in gradient
            
            # Safety: Avoid division by zero
            if np.dot(y, s) == 0: 
                break
                
            rho = 1.0 / np.dot(y, s)
            I = np.eye(dim)
            
            # The BFGS Update Formula (Sherman-Morrison)
            A = I - rho * np.outer(s, y)
            B = I - rho * np.outer(y, s)
            H_inv = (A @ H_inv @ B) + (rho * np.outer(s, s))
            
            params = params_new
            grad = grad_new
            
        return params
        
class NelderMead(Optimizer):
    """
    Gradient-Free Optimizer (Simplex Method).
    Uses geometric operations (Reflect, Expand, Contract, Shrink).
    """
    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, max_iters=1000, tol=1e-6):
        super().__init__(max_iters, tol)
        self.alpha = alpha # Reflection coefficient
        self.gamma = gamma # Expansion coefficient
        self.rho = rho     # Contraction coefficient
        self.sigma = sigma # Shrink coefficient

    def minimize(self, cost_function, initial_params, gradient_function=None, hessian_function=None):
        dim = len(initial_params)
        
        # 1. Initialize Simplex (N+1 points)
        simplex = [np.array(initial_params)]
        for i in range(dim):
            point = np.array(initial_params)
            point[i] = point[i] + 0.05 # Small perturbation
            simplex.append(point)
        simplex = np.array(simplex)
        
        for i in range(self.max_iters):
            # Sort vertices by cost
            costs = np.array([cost_function(p) for p in simplex])
            indices = np.argsort(costs)
            simplex = simplex[indices]
            costs = costs[indices]
            
            self.history.append({'params': simplex[0].copy().tolist(), 'cost': costs[0]})
            
            # Check Convergence (Simplex Size)
            if np.linalg.norm(simplex[0] - simplex[-1]) < self.tol:
                break

            best_point = simplex[0]
            worst_point = simplex[-1]
            
            # Centroid of the best N points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # A. REFLECT
            reflected = centroid + self.alpha * (centroid - worst_point)
            r_cost = cost_function(reflected)
            if costs[0] <= r_cost < costs[-2]:
                simplex[-1] = reflected
                continue
                
            # B. EXPAND
            if r_cost < costs[0]:
                expanded = centroid + self.gamma * (reflected - centroid)
                if cost_function(expanded) < r_cost: simplex[-1] = expanded
                else: simplex[-1] = reflected
                continue
                
            # C. CONTRACT
            if r_cost >= costs[-2]:
                if r_cost < costs[-1]: # Contract Outside
                    contracted = centroid + self.rho * (reflected - centroid)
                    if cost_function(contracted) < r_cost:
                        simplex[-1] = contracted
                        continue
                else: # Contract Inside
                    contracted = centroid + self.rho * (worst_point - centroid)
                    if cost_function(contracted) < costs[-1]:
                        simplex[-1] = contracted
                        continue

            # D. SHRINK
            new_simplex = [best_point]
            for p in simplex[1:]:
                new_simplex.append(best_point + self.sigma * (p - best_point))
            simplex = np.array(new_simplex)
            
        return simplex[0]

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam).
    Uses momentum (m) and adaptive learning rates (v) to handle stiff valleys.
    """
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iters=1000, tol=1e-6):
        super().__init__(max_iters, tol)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def minimize(self, cost_function, initial_params, gradient_function=None, hessian_function=None):
        params = np.array(initial_params)
        
        # Auto-detect gradient
        if gradient_function is None:
            gradient_function = lambda p: get_numerical_gradient(cost_function, p)
            
        # Initialize Moments
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 0
        
        for i in range(self.max_iters):
            t += 1
            cost = cost_function(params)
            grad = gradient_function(params)
            self.history.append({'params': params.copy().tolist(), 'cost': cost})
            
            # 1. Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad
            
            # 2. Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # 3. Compute bias-corrected first moment estimate
            m_hat = m / (1 - self.beta1 ** t)
            
            # 4. Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - self.beta2 ** t)
            
            # 5. Update parameters
            step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            new_params = params - step
            
            if np.linalg.norm(new_params - params) < self.tol:
                break
                
            params = new_params
            
        return params