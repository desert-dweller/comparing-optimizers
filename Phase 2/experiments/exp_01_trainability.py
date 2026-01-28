# Force numpy to use 1 core per process to avoid CPU oversubscription
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
DEPTHS = [2, 4, 8, 12]   
TRIALS = 10              
SHOTS = 1000             
QUBITS = 6               

def run_single_trial(task_conf):
    try:
        set_seed(task_conf['seed'])
        task = MaxCutTask(n_qubits=QUBITS, depth=task_conf['depth'], shots=SHOTS)
        cost_fn, n_params = task.get_cost_fn()
        
        init_params = np.random.uniform(0, 2*np.pi, n_params)
        
        opt_config = {'lr': 0.05, 'iters': 80, 'shots': SHOTS}
        
        if task_conf['opt'] == 'Adam':
            hist = Optimizers.adam(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'SPSA':
            hist = Optimizers.spsa(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'QuGStep':
            hist = Optimizers.qugstep(cost_fn, init_params, opt_config)
            
        final_params = hist[-1]
        final_cost = cost_fn(final_params)
        success = bool(final_cost < -4.0)
        
        return {
            **task_conf,
            'final_cost': float(final_cost),
            'success': success,
            'history': [float(cost_fn(p)) for p in hist[::5]] 
        }
    except Exception as e:
        return {'error': str(e), **task_conf}

def main():
    logger = ExperimentLogger("Exp01_Trainability")
    tasks = []
    
    # Generate Tasks
    for d in DEPTHS:
        for t in range(TRIALS):
            for opt in ['Adam', 'SPSA', 'QuGStep']:
                tasks.append({
                    'depth': d, 'trial': t, 'opt': opt, 
                    'seed': t*100 + d
                })
    
    logger.log_config({'depths': DEPTHS, 'trials': TRIALS, 'shots': SHOTS, 'qubits': QUBITS})
    print(f"[RUNNER] Starting Experiment 01: {len(tasks)} tasks running in parallel...")

    # Parallel Execution with Real-Time Updates
    results = []
    # Using all available cores (os.cpu_count())
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_single_trial, t): t for t in tasks}
        
        # Process as they finish (not in order)
        completed_count = 0
        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            completed_count += 1
            
            if completed_count % 5 == 0:
                logger.log_step(completed_count, info=f"Progress: {completed_count}/{len(tasks)} tasks done.")

    logger.metrics['results'] = results
    logger.finish(success_check=len(results) == len(tasks))

if __name__ == '__main__':
    main()