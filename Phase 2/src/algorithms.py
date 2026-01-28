# src/algorithms.py
import numpy as np

class Optimizers:
    """
    Standardized Optimizer Suite.
    All inputs: (cost_fn, init_params, config)
    All outputs: history (list of param arrays)
    """

    @staticmethod
    def _compute_finite_diff_grad(cost_fn, params, shift):
        """
        Helper: Calculates gradient via Central Difference.
        Cost: 2 * N_params function calls.
        """
        grad = np.zeros_like(params)
        
        # Iterate over each parameter
        for i in range(len(params)):
            # Create buffers to avoid cumulative errors
            p_plus = params.copy()
            p_minus = params.copy()
            
            # Shift
            p_plus[i] += shift
            p_minus[i] -= shift
            
            # Evaluate (Expensive step)
            loss_plus = cost_fn(p_plus)
            loss_minus = cost_fn(p_minus)
            
            # Central Difference Rule
            grad[i] = (loss_plus - loss_minus) / (2 * shift)
            
        return grad

    @staticmethod
    def adam(cost_fn, init_params, config):
        """
        Standard Adam with Fixed Finite Difference.
        Represents the 'Naive' Gradient Descent on Hardware.
        """
        p = np.array(init_params)
        lr = config.get('lr', 0.05)
        iters = config.get('iters', 100)
        # Fixed small shift (Standard approximation)
        shift = 0.01 
        
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        history = []

        for t in range(iters):
            # 1. Compute Gradient (Fixed Step)
            g = Optimizers._compute_finite_diff_grad(cost_fn, p, shift)
            
            # 2. Adam Update
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**(t + 1))
            v_hat = v / (1 - beta2**(t + 1))
            
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
            history.append(p.copy())
            
        return history

    @staticmethod
    def spsa(cost_fn, init_params, config):
        """
        SPSA: Simultaneous Perturbation Stochastic Approximation.
        The 'Efficient' Competitor (2 shots per step).
        """
        p = np.array(init_params)
        iters = config.get('iters', 100)
        # Fairness: SPSA is usually run for more iterations because it's cheaper.
        # We cap it here, but in analysis, we compare 'Total Shots'.
        
        a, c, alpha, gamma = 0.6, 0.1, 0.602, 0.101
        A = iters / 10
        history = []

        for k in range(iters):
            # Decay Schedules
            ak = a / (k + 1 + A)**alpha
            ck = c / (k + 1)**gamma
            
            # Random Perturbation Vector
            delta = np.random.choice([-1, 1], size=p.shape)
            
            # 2 Function Evaluations (Cheap!)
            y_plus = cost_fn(p + ck * delta)
            y_minus = cost_fn(p - ck * delta)
            
            # Gradient Estimate
            ghat = (y_plus - y_minus) / (2 * ck * delta)
            
            # Update
            p -= ak * ghat
            history.append(p.copy())
            
        return history

    @staticmethod
    def qugstep(cost_fn, init_params, config):
        """
        QuGStep: Noise-Adaptive Gradient Descent.
        Innovation: Shift epsilon scales with shot noise S^(-1/4).
        """
        p = np.array(init_params)
        lr = config.get('lr', 0.05)
        iters = config.get('iters', 100)
        shots = config.get('shots', 1000)
        
        # --- THE ALGORITHM CORE ---
        # Scaling constant 1.5 is empirical but standard for this topology
        adaptive_shift = 1.5 * (shots ** -0.25)
        
        history = []
        
        for t in range(iters):
            # 1. Compute Gradient (Adaptive Step)
            g = Optimizers._compute_finite_diff_grad(cost_fn, p, adaptive_shift)
            
            # 2. Descent Update (Vanilla Gradient Descent to isolate effect)
            # (Can also use Momentum, but Vanilla proves the gradient quality better)
            p -= lr * g
            history.append(p.copy())
            
        return history