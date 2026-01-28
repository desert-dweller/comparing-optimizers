import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problems import MakeMoonsTask
from src.algorithms import Optimizers
from src.utils import ExperimentLogger, set_seed

# --- CONFIGURATION ---
TRIALS = 10
DEPTH = 4       
SHOTS = 1000
SAMPLES = 60    

def run_single_trial(task_conf):
    try:
        set_seed(task_conf['seed'])
        
        task = MakeMoonsTask(n_samples=SAMPLES, noise=0.1)
        loss_fn, n_params = task.get_loss_fn(n_qubits=4, depth=DEPTH, shots=SHOTS)
        init_params = np.random.uniform(0, 2*np.pi, n_params)
        
        opt_config = {'lr': 0.1, 'iters': 40, 'shots': SHOTS}
        
        if task_conf['opt'] == 'Adam':
            hist = Optimizers.adam(loss_fn, init_params, opt_config)
        elif task_conf['opt'] == 'SPSA':
            hist = Optimizers.spsa(loss_fn, init_params, opt_config)
        elif task_conf['opt'] == 'QuGStep':
            hist = Optimizers.qugstep(loss_fn, init_params, opt_config)
        
        # Test Generalization (Fresh Data)
        test_task = MakeMoonsTask(n_samples=100, noise=0.1)
        test_loss_fn, _ = test_task.get_loss_fn(n_qubits=4, depth=DEPTH, shots=SHOTS)
        final_test_loss = test_loss_fn(hist[-1])
        accuracy_proxy = 1.0 - final_test_loss 
        
        return {
            **task_conf,
            'final_train_loss': float(loss_fn(hist[-1])),
            'final_test_loss': float(final_test_loss),
            'accuracy_score': float(accuracy_proxy)
        }
    except Exception as e:
        return {'error': str(e), **task_conf}

def main():
    logger = ExperimentLogger("Exp03_QML_Utility")
    tasks = []
    
    for t in range(TRIALS):
        for opt in ['Adam', 'SPSA', 'QuGStep']:
            tasks.append({'trial': t, 'opt': opt, 'seed': t*555})

    logger.log_config({'depth': DEPTH, 'samples': SAMPLES})
    print(f"[RUNNER] Starting Experiment 03: {len(tasks)} tasks running in parallel...")

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