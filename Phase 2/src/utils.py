import numpy as np
import json
import os
import time
import logging
import uuid
from datetime import datetime

class ExperimentLogger:
    """
    The 'Black Box' recorder.
    Saves metrics, configuration, and logs to a unique folder per run.
    """
    def __init__(self, exp_name, output_dir="results"):
        # 1. Create Unique Run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{exp_name}_{timestamp}_{str(uuid.uuid4())[:4]}"
        self.exp_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 2. Setup Python Logging
        self.logger = logging.getLogger(self.run_id)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # File Handler (UTF-8 Enforced for Windows)
        fh = logging.FileHandler(os.path.join(self.exp_dir, "experiment.log"), encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)
        
        self.logger.info(f"[START] Experiment Initialized: {self.run_id}")
        
        # 3. Data Storage
        self.metrics = {
            'history': [],
            'final_summary': {},
            'config': {}
        }
        self.start_time = time.time()

    def log_config(self, config_dict):
        """Save hyperparameters."""
        self.metrics['config'] = config_dict
        self.logger.info(f"[CONFIG] Configuration: {json.dumps(config_dict, indent=2)}")

    def log_step(self, step, **kwargs):
        """
        Record a single optimization step.
        """
        entry = {'step': step, 'timestamp': time.time() - self.start_time}
        entry.update(kwargs)
        self.metrics['history'].append(entry)
        
        if 'info' in kwargs:
            self.logger.info(f"Step {step}: {kwargs['info']}")

    def finish(self, success_check=None):
        """
        Save all data to JSON.
        """
        duration = time.time() - self.start_time
        self.metrics['final_summary']['duration'] = duration
        
        if success_check is not None:
            self.metrics['final_summary']['success'] = success_check
            self.logger.info(f"[DONE] Run Complete. Success: {success_check}")
        else:
            self.logger.info("[DONE] Run Complete.")

        # Save to JSON
        json_path = os.path.join(self.exp_dir, "metrics.json")
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4, cls=NumpyEncoder)
        
        self.logger.info(f"[SAVED] Results saved to: {json_path}")
        return json_path

def set_seed(seed):
    """Ensure Reproducibility."""
    np.random.seed(seed)