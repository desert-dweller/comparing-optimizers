import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problems import MakeMoonsTask  # <--- CHANGED
from src.algorithms import Optimizers
from src.utils import ExperimentLogger, set_seed

# --- PRODUCTION CONFIG ---
TRIALS = 50           # High statistical significance for Violin Plot
DEPTH = 4             # Standard for Moons (L=4 in paper)
SHOTS = 1000          # Standard NISQ Noise
QUBITS = 2            # 2 Features = 2 Qubits
N_SAMPLES = 100       # Matches Paper Section 3.4

def run_single_trial(task_conf):
    try:
        set_seed(task_conf['seed'])
        
        # Initialize Task: Make Moons
        task = MakeMoonsTask(n_qubits=QUBITS, n_samples=N_SAMPLES, seed=task_conf['seed'])
        
        loss_fn, n_params = task.get_loss_fn(n_qubits=QUBITS, depth=DEPTH, shots=SHOTS)
        init_params = np.random.uniform(0, 2*np.pi, n_params)
        
        # QML Config
        opt_config = {'lr': 0.1, 'iters': 60, 'shots': SHOTS}
        
        if task_conf['opt'] == 'Adam': 
            hist = Optimizers.adam(loss_fn, init_params, opt_config)
        elif task_conf['opt'] == 'SPSA': 
            hist = Optimizers.spsa(loss_fn, init_params, opt_config)
        elif task_conf['opt'] == 'QuGStep': 
            hist = Optimizers.qugstep(loss_fn, init_params, opt_config)
        
        # Evaluate
        acc = task.get_test_accuracy(hist[-1], n_qubits=QUBITS, depth=DEPTH, shots=SHOTS)
        final_loss = loss_fn(hist[-1])
        
        return {
            **task_conf, 
            'accuracy_score': float(acc),
            'final_train_loss': float(final_loss),
            'nfe': loss_fn.calls
        }
        
    except Exception as e:
        return {'error': str(e), **task_conf}

def main():
    logger = ExperimentLogger("Exp03_MakeMoons_Production")
    
    tasks = []
    for t in range(TRIALS):
        for opt in ['Adam', 'SPSA', 'QuGStep']:
            tasks.append({'trial': t, 'opt': opt, 'seed': t*777 + 55})

    max_workers = max(1, (os.cpu_count() or 4) - 2)
    
    logger.log_config({'trials': TRIALS, 'dataset': 'Make Moons', 'qubits': QUBITS})
    print(f"[RUNNER] Starting PRODUCTION Exp 03 (Moons): {len(tasks)} tasks...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_single_trial, t): t for t in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 10 == 0: 
                logger.log_step(completed, info=f"Progress: {completed}/{len(tasks)}")

    logger.metrics['results'] = results
    logger.finish()

if __name__ == '__main__':
    main()