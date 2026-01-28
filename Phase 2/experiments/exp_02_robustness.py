import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problems import MaxCutTask
from src.algorithms import Optimizers
from src.utils import ExperimentLogger, set_seed

# --- CONFIGURATION ---
SHOT_LEVELS = [100, 500, 1000, 5000]
FIXED_DEPTH = 8
TRIALS = 10
QUBITS = 6

def run_single_trial(task_conf):
    try:
        set_seed(task_conf['seed'])
        shots = task_conf['shots']
        
        task = MaxCutTask(n_qubits=QUBITS, depth=FIXED_DEPTH, shots=shots)
        cost_fn, n_params = task.get_cost_fn()
        init_params = np.random.uniform(0, 2*np.pi, n_params)
        
        opt_config = {'lr': 0.05, 'iters': 80, 'shots': shots}
        
        if task_conf['opt'] == 'Adam':
            hist = Optimizers.adam(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'SPSA':
            hist = Optimizers.spsa(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'QuGStep':
            hist = Optimizers.qugstep(cost_fn, init_params, opt_config)
            
        final_cost = cost_fn(hist[-1])
        
        return {**task_conf, 'final_cost': float(final_cost)}
    except Exception as e:
        return {'error': str(e), **task_conf}

def main():
    logger = ExperimentLogger("Exp02_Robustness")
    tasks = []
    
    for s in SHOT_LEVELS:
        for t in range(TRIALS):
            for opt in ['Adam', 'SPSA', 'QuGStep']:
                tasks.append({
                    'shots': s, 'trial': t, 'opt': opt, 
                    'seed': t*999 + s
                })

    logger.log_config({'shot_levels': SHOT_LEVELS, 'fixed_depth': FIXED_DEPTH})
    print(f"[RUNNER] Starting Experiment 02: {len(tasks)} tasks running in parallel...")

    results = []
    with ProcessPoolExecutor() as executor:
        future_to_task = {executor.submit(run_single_trial, t): t for t in tasks}
        
        completed_count = 0
        for future in as_completed(future_to_task):
            results.append(future.result())
            completed_count += 1
            if completed_count % 5 == 0:
                logger.log_step(completed_count, info=f"Progress: {completed_count}/{len(tasks)}")
            
    logger.metrics['results'] = results
    logger.finish()

if __name__ == '__main__':
    main()