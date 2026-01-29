import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems import MaxCutRegularTask
from src.algorithms import Optimizers
from src.utils import ExperimentLogger, set_seed

# --- PRODUCTION CONFIG ---
DEPTHS = [2, 4, 6, 8, 10, 12] 
TRIALS = 30 
SHOTS = 1000
QUBITS = 8 

def run_single_trial(task_conf):
    try:
        set_seed(task_conf['seed'])
        task = MaxCutRegularTask(n_qubits=QUBITS, depth=task_conf['depth'], shots=SHOTS, seed=task_conf['seed'])
        cost_fn, n_params = task.get_cost_fn()
        init_params = np.random.uniform(0, 2*np.pi, n_params)
        
        opt_config = {'lr': 0.05, 'iters': 100, 'shots': SHOTS}
        
        if task_conf['opt'] == 'Adam': hist = Optimizers.adam(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'SPSA': hist = Optimizers.spsa(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'QuGStep': hist = Optimizers.qugstep(cost_fn, init_params, opt_config)
            
        final_cost = cost_fn(hist[-1])
        
        # Calculate Convergence Speed
        convergence_step = 100
        for i, params in enumerate(hist):
            if cost_fn(params) < -3.0:
                convergence_step = i
                break

        return {
            **task_conf,
            'final_cost': float(final_cost),
            'nfe': cost_fn.calls,
            'convergence_step': int(convergence_step),
            'history': [float(cost_fn(p)) for p in hist[::10]]
        }
    except Exception as e:
        return {'error': str(e), **task_conf}

def main():
    logger = ExperimentLogger("Exp01_Production")
    tasks = []
    for d in DEPTHS:
        for t in range(TRIALS):
            for opt in ['Adam', 'SPSA', 'QuGStep']:
                tasks.append({'depth': d, 'trial': t, 'opt': opt, 'seed': t*100 + d})
    
    # THROTTLE PARALLELISM
    max_workers = max(1, (os.cpu_count() or 4) - 2)
    logger.log_config({'depths': DEPTHS, 'trials': TRIALS, 'metrics': ['nfe', 'convergence']})
    print(f"[RUNNER] Exp 01: {len(tasks)} tasks (Workers: {max_workers})...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_single_trial, t): t for t in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            results.append(future.result())
            completed += 1
            if completed % 10 == 0: logger.log_step(completed, info=f"Progress: {completed}/{len(tasks)}")
            
    logger.metrics['results'] = results
    logger.finish()

if __name__ == '__main__':
    main()