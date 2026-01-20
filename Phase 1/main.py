import json
import datetime
import csv
import os
import numpy as np
import torch
from scipy.optimize import minimize
from optimizers import Adam, NelderMead, NewtonMethod, BFGS, get_numerical_gradient, get_numerical_hessian
from benchmarks import test_cases

# --- CONFIGURATION ---
MAX_ITERATIONS = 200
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- FOLDERS ---
LOG_DIR = "logs"
SUMMARY_DIR = "results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

JSON_FILE = os.path.join(LOG_DIR, f"benchmark_data_{TIMESTAMP}.json")
CSV_FILE = os.path.join(SUMMARY_DIR, f"benchmark_summary_{TIMESTAMP}.csv")

# --- RUNNER: CUSTOM ---
def run_custom(optimizer_class, cost_func, start_params, **kwargs):
    opt = optimizer_class(max_iters=MAX_ITERATIONS, **kwargs)
    res_params = opt.minimize(cost_func, start_params)
    return {
        'final_cost': cost_func(res_params),
        'iterations': len(opt.history),
        'history': opt.history 
    }

# --- RUNNER: PYTORCH ---
def run_pytorch_adam(cost_func, start_params, learning_rate=0.1):
    params = torch.tensor(start_params, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([params], lr=learning_rate)
    history = []
    
    for i in range(MAX_ITERATIONS):
        p_numpy = params.detach().numpy()
        current_cost = cost_func(p_numpy)
        history.append({'params': p_numpy.copy().tolist(), 'cost': current_cost})
        
        num_grad = get_numerical_gradient(cost_func, p_numpy)
        optimizer.zero_grad()
        params.grad = torch.tensor(num_grad, dtype=torch.float64)
        optimizer.step()
        
        if i > 0 and np.linalg.norm(p_numpy - history[-2]['params']) < 1e-6:
            break
                
    return {
        'final_cost': history[-1]['cost'],
        'iterations': len(history),
        'history': history
    }

# --- RUNNER: SCIPY ---
def run_scipy(method, cost_func, start_params):
    history = []
    def callback(xk):
        history.append({'params': xk.tolist(), 'cost': cost_func(xk)})
    
    grad = lambda p: get_numerical_gradient(cost_func, p)
    hess = lambda p: get_numerical_hessian(cost_func, p)
    
    # Smart Argument Handling: Only pass gradient/hessian if the method uses it
    use_jac = grad if method in ['Newton-CG', 'BFGS', 'L-BFGS-B'] else None
    use_hess = hess if method == 'Newton-CG' else None
    
    try:
        res = minimize(cost_func, start_params, method=method, jac=use_jac, hess=use_hess, 
                       callback=callback, options={'maxiter': MAX_ITERATIONS})
    except Exception as e:
        print(f"  SciPy {method} failed: {e}")
        return {'final_cost': float('inf'), 'iterations': 0, 'history': []}

    if not history: 
        history.append({'params': res.x.tolist(), 'cost': res.fun})
        
    return {
        'final_cost': res.fun,
        'iterations': res.nit,
        'history': history
    }

# --- EXECUTION LOOP ---
if __name__ == "__main__":
    full_results = {}
    print(f"🚀 Starting Experiment Run ID: {TIMESTAMP}")

    for case in test_cases:
        print(f"\n--- Testing: {case.name} ---")
        case_results = {}
        
        # 1. Adam Comparison
        print("  Running Custom Adam...")
        case_results['Custom Adam'] = run_custom(Adam, case.func, case.start_params, learning_rate=0.1)
        print("  Running PyTorch Adam...")
        case_results['PyTorch Adam'] = run_pytorch_adam(case.func, case.start_params, learning_rate=0.1)

        # 2. Newton Comparison
        print("  Running Custom Newton...")
        case_results['Custom Newton'] = run_custom(NewtonMethod, case.func, case.start_params)
        print("  Running SciPy Newton...")
        case_results['SciPy Newton'] = run_scipy('Newton-CG', case.func, case.start_params)

        # 3. BFGS Comparison
        print("  Running Custom BFGS...")
        case_results['Custom BFGS'] = run_custom(BFGS, case.func, case.start_params)
        print("  Running SciPy BFGS...")
        case_results['SciPy BFGS'] = run_scipy('BFGS', case.func, case.start_params)
        
        # 4. Nelder-Mead Comparison (NEW!)
        print("  Running Custom NM...")
        case_results['Custom NM'] = run_custom(NelderMead, case.func, case.start_params)
        print("  Running SciPy NM...")
        case_results['SciPy NM'] = run_scipy('Nelder-Mead', case.func, case.start_params)

        full_results[case.name] = case_results

    # --- SAVE ---
    with open(JSON_FILE, 'w') as f:
        json.dump(full_results, f, indent=4)
    print(f"\n💾 Log saved to: {JSON_FILE}")

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Function", "Optimizer", "Final Cost", "Iterations"])
        for func_name, func_data in full_results.items():
            for opt_name, metrics in func_data.items():
                writer.writerow([func_name, opt_name, f"{metrics['final_cost']:.4e}", metrics['iterations']])
    print(f"📄 Summary saved to: {CSV_FILE}")