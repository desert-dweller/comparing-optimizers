import numpy as np
import json
import datetime
import os
from quantum_interface import QuantumObjective
# Import your ORIGINAL optimizers from Phase 1
from optimizers import Adam, BFGS 

# --- CONFIG ---
DEPTHS_TO_TEST = [2, 5, 10, 15] # The "Barren Plateau" Ladder
SHOTS = 1000                    # Noisy regime
PROBLEM = 'COMPILER'            # 'COMPILER' Or 'QML'
MAX_ITER = 150
RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

results = {}

print(f"🧪 Starting Phase 2 Benchmark: {PROBLEM} (Shots={SHOTS})")

for depth in DEPTHS_TO_TEST:
    print(f"\n--- Testing Depth L={depth} ---")
    
    # 1. Initialize the Quantum Environment
    obj = QuantumObjective(num_qubits=3, num_layers=depth, problem_type=PROBLEM, n_shots=SHOTS)
    n_params = obj.get_num_params()
    
    # Random Initialization (Start in the plateau)
    start_params = np.random.uniform(0, 2*np.pi, n_params)
    
    depth_results = {}

    # --- RUN 1: Standard Adam (Fixed Epsilon) ---
    # By passing gradient_function=None, your optimizer uses the default 1e-8 step
    print("  1. Running Standard Adam (Fixed Epsilon)...")
    opt_std = Adam(max_iters=MAX_ITER, learning_rate=0.05)
    res_std = opt_std.minimize(obj.func, start_params, gradient_function=None) 
    
    depth_results['Standard_Adam'] = {
        'final_cost': obj.func(res_std),
        'history': opt_std.history
    }

    # --- RUN 2: Research Adam (QuGStep) ---
    # Here we inject the new Adaptive Gradient function
    print("  2. Running QuGStep Adam (Adaptive Epsilon)...")
    opt_res = Adam(max_iters=MAX_ITER, learning_rate=0.05)
    res_res = opt_res.minimize(obj.func, start_params, gradient_function=obj.get_adaptive_gradient)
    
    depth_results['QuGStep_Adam'] = {
        'final_cost': obj.func(res_res),
        'history': opt_res.history
    }
    
    results[f"Depth_{depth}"] = depth_results

# --- SAVE RESULTS ---
os.makedirs("results_phase2", exist_ok=True)
filename = f"results_phase2/benchmark_{PROBLEM}_{RUN_ID}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Results saved to {filename}")