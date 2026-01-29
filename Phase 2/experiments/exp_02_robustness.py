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
SHOT_LEVELS = [100, 200, 500, 1000, 2000, 5000, 10000]
FIXED_DEPTH = 8
TRIALS = 30
QUBITS = 8

def run_single_trial(task_conf):
    try:
        set_seed(task_conf['seed'])
        shots = task_conf['shots']
        task = MaxCutRegularTask(n_qubits=QUBITS, depth=FIXED_DEPTH, shots=shots)
        cost_fn, n_params = task.get_cost_fn()
        init_params = np.random.uniform(0, 2*np.pi, n_params)
        
        opt_config = {'lr': 0.05, 'iters': 80, 'shots': shots}
        
        if task_conf['opt'] == 'Adam': hist = Optimizers.adam(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'SPSA': hist = Optimizers.spsa(cost_fn, init_params, opt_config)
        elif task_conf['opt'] == 'QuGStep': hist = Optimizers.qugstep(cost_fn, init_params, opt_config)
            
        return {
            **task_conf, 
            'final_cost': float(cost_fn(hist[-1])),
            'nfe': cost_fn.calls
        }
    except Exception as e:
        return {'error': str(e), **task_conf}

def main():
    logger = ExperimentLogger("Exp02_Production")
    tasks = []
    for s in SHOT_LEVELS:
        for t in range(TRIALS):
            for opt in ['Adam', 'SPSA', 'QuGStep']:
                tasks.append({'shots': s, 'trial': t, 'opt': opt, 'seed': t*999 + s})

    max_workers = max(1, (os.cpu_count() or 4) - 2)
    logger.log_config({'shot_levels': SHOT_LEVELS, 'trials': TRIALS})
    print(f"[RUNNER] Exp 02: {len(tasks)} tasks...")

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