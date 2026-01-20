import numpy as np

# 1. Define the "Complex" Function
def my_complex_function(x):
    # f(x) = x^2 * sin(x) + e^(-x)
    return (x**2 * np.sin(x)) + np.exp(-x)

# 2. Define the Exact Derivative (for comparison only)
def exact_derivative(x):
    # Using Product Rule and Chain Rule
    term1 = 2*x * np.sin(x) + x**2 * np.cos(x)
    term2 = -np.exp(-x)
    return term1 + term2

# 3. The Numerical Method (Finite Difference)
def get_gradient_numerically(func, x, epsilon=1e-6):
    """
    The computer doesn't know calculus rules.
    It just measures the slope between x-e and x+e.
    """
    y_forward = func(x + epsilon)
    y_backward = func(x - epsilon)
    
    rise = y_forward - y_backward
    run = 2 * epsilon
    
    return rise / run

# --- Run the Experiment ---

target_x = 1.5  # Let's check the slope at x = 1.5

# Calculate numerically
numeric_result = get_gradient_numerically(my_complex_function, target_x)

# Calculate exactly
exact_result = exact_derivative(target_x)

print(f"Function:           f(x) = x^2 * sin(x) + e^(-x)")
print(f"Target Point:       x = {target_x}")
print("-" * 50)
print(f"Numerical Gradient: {numeric_result:.8f}")
print(f"Exact Gradient:     {exact_result:.8f}")
print("-" * 50)
print(f"Difference (Error): {abs(numeric_result - exact_result):.10f}")