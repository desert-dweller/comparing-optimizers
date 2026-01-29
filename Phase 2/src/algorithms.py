import numpy as np

class Optimizers:
    """
    Standardized Optimizer Suite.
    Ensures 'Apples-to-Apples' comparison by using Finite Differences for all gradient methods.
    """

    @staticmethod
    def _compute_finite_diff_grad(cost_fn, params, shift):
        """
        Calculates gradient via Central Difference.
        Cost: 2 * N_params function calls.
        """
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += shift
            p_minus[i] -= shift
            
            # These calls increment the CostTracker
            loss_plus = cost_fn(p_plus)
            loss_minus = cost_fn(p_minus)
            
            grad[i] = (loss_plus - loss_minus) / (2 * shift)
        return grad

    @staticmethod
    def adam(cost_fn, init_params, config):
        """Standard Adam (Fixed Shift = 0.01)."""
        p = np.array(init_params)
        lr = config.get('lr', 0.05)
        iters = config.get('iters', 100)
        shift = 0.01 
        
        m = np.zeros_like(p); v = np.zeros_like(p)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        history = []

        for t in range(iters):
            g = Optimizers._compute_finite_diff_grad(cost_fn, p, shift)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**(t + 1))
            v_hat = v / (1 - beta2**(t + 1))
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
            history.append(p.copy())
        return history

    @staticmethod
    def spsa(cost_fn, init_params, config):
        """SPSA (2 calls per step). Efficient but Noisy."""
        p = np.array(init_params)
        iters = config.get('iters', 100)
        a, c, alpha, gamma = 0.6, 0.1, 0.602, 0.101
        A = iters / 10
        history = []

        for k in range(iters):
            ak = a / (k + 1 + A)**alpha
            ck = c / (k + 1)**gamma
            delta = np.random.choice([-1, 1], size=p.shape)
            
            y_plus = cost_fn(p + ck * delta)
            y_minus = cost_fn(p - ck * delta)
            
            ghat = (y_plus - y_minus) / (2 * ck * delta)
            p -= ak * ghat
            history.append(p.copy())
        return history

    @staticmethod
    def qugstep(cost_fn, init_params, config):
        """QuGStep (Adaptive Shift). Scales with Noise."""
        p = np.array(init_params)
        lr = config.get('lr', 0.05)
        iters = config.get('iters', 100)
        shots = config.get('shots', 1000)
        
        # Scaling Law: epsilon scales with noise
        adaptive_shift = 1.5 * (shots ** -0.25)
        
        history = []
        for t in range(iters):
            g = Optimizers._compute_finite_diff_grad(cost_fn, p, adaptive_shift)
            p -= lr * g
            history.append(p.copy())
        return history