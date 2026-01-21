import numpy as np
import json
import datetime
import os
import torch
from scipy.optimize import minimize as scipy_minimize
from joblib import Parallel, delayed  # The Parallel Engine

# --- IMPORTS ---
from quantum_interface import QuantumObjective
from optimizers import *

# Check for Qiskit Algorithms
try:
    from qiskit_algorithms.optimizers import SPSA as QiskitSPSA
    HAS_QISKIT_ALGOS = True
except ImportError:
    HAS_QISKIT_ALGOS = False

# --- CONFIG ---
DEPTHS_TO_TEST = [2, 5, 10]     # These run SIMULTANEOUSLY
SHOTS = 1000                    # Noise Level
PROBLEM = 'QML'                 # CHANGE TO 'COMPILER' IN YOUR 2ND TERMINAL
MAX_ITER = 150
RUN_ID = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
N_JOBS = -1                     # -1 = Use ALL CPU Cores

# --- WRAPPERS ---
def run_pytorch(obj, start, iters):
    # Important: Prevent PyTorch from spawning threads inside a parallel worker
    torch.set_num_threads(1) 
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.05)
    hist = []
    for _ in range(iters):
        cost = obj.func(p.detach().numpy())
        hist.append({'cost': cost})
        grad = obj.get_adaptive_gradient(p.detach().numpy())
        opt.zero_grad(); p.grad = torch.tensor(grad, dtype=torch.float64); opt.step()
    return {'final_cost': hist[-1]['cost'], 'history': hist}

def run_scipy(obj, start, method, iters, use_grad=False):
    hist = []
    def log(x): hist.append({'cost': obj.func(x)})
    jac = obj.get_adaptive_gradient if use_grad else None
    res = scipy_minimize(lambda x: obj.func(x), start, method=method, jac=jac, options={'maxiter': iters}, callback=log)
    if not hist: hist.append({'cost': res.fun})
    return {'final_cost': res.fun, 'history': hist}

# --- PARALLEL WORKER FUNCTION ---
def run_benchmark_for_depth(depth, problem, shots, max_iter):
    print(f"   ⚡ Starting Worker for Depth {depth}...")
    
    # Re-initialize object inside the worker process (Crucial for Parallel Safety)
    obj = QuantumObjective(num_qubits=3, num_layers=depth, problem_type=problem, n_shots=shots, gate_error_rate=0.005)
    start = np.random.uniform(0, 2*np.pi, obj.get_num_params())
    d_res = {}

    # 1. BASELINES
    d_res['1_NelderMead_Custom'] = {'final_cost': obj.func(NelderMead(max_iter).minimize(obj.func, start)), 'history': []}
    if HAS_QISKIT_ALGOS:
        spsa = QiskitSPSA(maxiter=max_iter)
        res = spsa.minimize(fun=obj.func, x0=start)
        d_res['2_SPSA_Qiskit'] = {'final_cost': res.fun, 'history': [{'cost': res.fun}]}

    # 2. COMPETITORS
    d_res['3_Adam_PyTorch'] = run_pytorch(obj, start, max_iter)
    
    opt = Adam(max_iters=max_iter, learning_rate=0.05)
    res = opt.minimize(obj.func, start, gradient_function=obj.get_adaptive_gradient)
    d_res['4_Adam_QuGStep'] = {'final_cost': obj.func(res), 'history': opt.history}

    d_res['5_BFGS_SciPy'] = run_scipy(obj, start, 'L-BFGS-B', max_iter, use_grad=True)
    
    opt = BFGS(max_iters=max_iter)
    res = opt.minimize(obj.func, start, gradient_function=obj.get_adaptive_gradient)
    d_res['6_BFGS_QuGStep'] = {'final_cost': obj.func(res), 'history': opt.history}

    # 3. RESEARCH GAPS
    opt = HybridOptimizer(switch_iter=40, max_iters=max_iter)
    res = opt.minimize(obj.func, start, gradient_function=obj.get_adaptive_gradient)
    d_res['7_Hybrid_Ours'] = {'final_cost': obj.func(res), 'history': opt.history}

    opt = SaddleFreeNewton(max_iters=max_iter)
    res = opt.minimize(obj.func, start, gradient_function=obj.get_adaptive_gradient)
    d_res['8_SaddleFree_Newton'] = {'final_cost': obj.func(res), 'history': opt.history}

    print(f"   ✅ Depth {depth} Finished!")
    return (f"Depth_{depth}", d_res)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 PARALLEL Speed-Run: {PROBLEM} (Cores Used: MAX)")
    
    # Run Depths in Parallel
    results_list = Parallel(n_jobs=N_JOBS)(
        delayed(run_benchmark_for_depth)(d, PROBLEM, SHOTS, MAX_ITER) 
        for d in DEPTHS_TO_TEST
    )
    
    results = dict(results_list)

    # --- SAVE ---
    os.makedirs("results_phase2", exist_ok=True)
    filename = f"results_phase2/FINAL_BENCHMARK_{PROBLEM}_{RUN_ID}.json"
    with open(filename, 'w') as f: json.dump(results, f, indent=4)
    print(f"\n✅ MASTER LOG SAVED: {filename}")