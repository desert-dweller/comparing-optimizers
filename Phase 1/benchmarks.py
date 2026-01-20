import numpy as np

class Benchmark:
    def __init__(self, name, func, start_params):
        self.name = name
        self.func = func
        self.start_params = np.array(start_params)

# --- 1. Sphere (Easy & Isotropic) ---
def sphere(p):
    return p[0]**2 + p[1]**2

# --- 2. Rosenbrock (Curved) ---
def rosenbrock(p):
    return (1.0 - p[0])**2 + 100.0 * (p[1] - p[0]**2)**2

# --- 3. Himmelblau (Multi-Modal / 4 Minima) ---
def himmelblau(p):
    return (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2

# --- 4. Ellipse (Steep Y, Flat X) ---
def ellipse(p):
    return p[0]**2 + 10 * p[1]**2

# --- 5. Booth (Plate-Shaped Valley) ---
# Min at (1, 3)
def booth(p):
    return (p[0] + 2*p[1] - 7)**2 + (2*p[0] + p[1] - 5)**2

# --- 6. Matyas (Rotated Bowl) ---
# Min at (0, 0). Tests parameter correlation (xy term).
def matyas(p):
    return 0.26 * (p[0]**2 + p[1]**2) - 0.48 * p[0] * p[1]

# --- 7. Three-Hump Camel (3 Minima) ---
# Min at (0, 0). Local minima at (+-1.7475, -0.8732)
def three_hump(p):
    return 2*p[0]**2 - 1.05*p[0]**4 + (p[0]**6)/6 + p[0]*p[1] + p[1]**2

# --- The Suite ---
test_cases = [
    Benchmark("Sphere", sphere, [-3.0, 3.0]),
    Benchmark("Rosenbrock", rosenbrock, [-1.2, 1.0]),
    Benchmark("Himmelblau", himmelblau, [0.0, 0.0]),
    Benchmark("Ellipse", ellipse, [-4.0, 3.0]),
    Benchmark("Booth", booth, [-5.0, 5.0]),
    Benchmark("Matyas", matyas, [5.0, 5.0]),
    Benchmark("ThreeHump", three_hump, [-2.0, 2.0])
]