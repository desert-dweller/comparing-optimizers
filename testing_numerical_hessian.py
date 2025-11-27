import numpy as np

# --- 1. Define the Test Function (Rosenbrock) ---
def simple_cost(params):
    return (1.0 - params[0])**2 + 100.0 * (params[1] - params[0]**2)**2

# --- 2. Define the EXACT Analytical Hessian ---
# We derived this via calculus manually
def get_analytical_hessian(params):
    x0, x1 = params[0], params[1]
    
    d2f_dx0_d0 = 2.0 - 400.0 * x1 + 1200.0 * x0**2
    d2f_dx1_d1 = 200.0
    d2f_dx0_d1 = -400.0 * x0
    
    return np.array([
        [d2f_dx0_d0, d2f_dx0_d1], 
        [d2f_dx0_d1, d2f_dx1_d1]
    ])

# --- 3. Define the NUMERICAL Implementation ---

def get_numerical_gradient(cost_func, params, epsilon=1e-5):
    """Calculates gradient via Finite Difference"""
    grad = np.zeros_like(params)
    params_copy = params.copy()
    
    for i in range(len(params)):
        original_val = params_copy[i]
        
        # Step Forward & Backward
        params_copy[i] = original_val + epsilon
        cost_plus = cost_func(params_copy)
        params_copy[i] = original_val - epsilon
        cost_minus = cost_func(params_copy)
        
        # Slope
        grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
        params_copy[i] = original_val # Reset
        
    return grad

def get_numerical_hessian(cost_func, params, epsilon=1e-4):
    """
    Calculates Hessian by taking the Finite Difference of the Gradient.
    Note: We use a larger epsilon (1e-4) for 2nd derivatives to avoid noise.
    """
    n = len(params)
    hessian = np.zeros((n, n))
    
    # We differentiate the gradient vector with respect to each parameter
    for i in range(n):
        params_copy = params.copy()
        
        # 1. Perturb parameter i forward
        params_copy[i] += epsilon
        grad_plus = get_numerical_gradient(cost_func, params_copy)
        
        # 2. Perturb parameter i backward
        params_copy[i] -= 2 * epsilon # (Go back to original - epsilon)
        grad_minus = get_numerical_gradient(cost_func, params_copy)
        
        # Reset
        params_copy[i] += epsilon 
        
        # 3. The change in gradient divided by distance is the 2nd derivative
        # This fills the i-th COLUMN of the Hessian
        hessian[:, i] = (grad_plus - grad_minus) / (2 * epsilon)
        
    return hessian

# --- 4. Compare Them ---

# Test Point: The standard starting point for Rosenbrock
test_params = np.array([-1.2, 1.0])

print(f"Testing Hessian at point: {test_params}\n")

# A. Analytical
exact_h = get_analytical_hessian(test_params)
print("--- Exact Analytical Hessian ---")
print(exact_h)
print()

# B. Numerical
num_h = get_numerical_hessian(simple_cost, test_params)
print("--- Numerical Hessian (Auto-Calculated) ---")
print(num_h)
print()

# rtol=1e-4 means "within 0.01% error"
if np.allclose(exact_h, num_h, rtol=1e-4, atol=1e-2):
    print("✅ SUCCESS: The numerical Hessian matches the analytical one!")
else:
    print("❌ FAILURE: The difference is too large.")