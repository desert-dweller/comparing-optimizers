# Breaking the Noise Barrier: QuGStep for NISQ Optimization


## Overview
This repository contains the source code, experimental drivers, and analysis scripts for **QuGStep** (Quantum Gradient Step-Scaling). QuGStep is a noise-adaptive optimization protocol designed to restore trainability in deep Variational Quantum Algorithms (VQAs) running on Noisy Intermediate-Scale Quantum (NISQ) devices.

The core hypothesis of this work is that standard gradient descent fails on NISQ hardware not merely due to the vanishing gradient problem (Barren Plateaus), but due to the signal-to-noise ratio crisis. We demonstrate that by dynamically scaling the finite-difference stencil size ($\epsilon$) inversely with the shot noise ($\epsilon \propto S^{-1/4}$), the gradient estimator functions as a **spatial low-pass filter**. This smooths out high-frequency statistical noise to reveal the global descent direction.

## Repository Structure

The project follows a modular structure designed for reproducibility:

```text
├── analysis/               # Visualization and Analysis
│   ├── plot_results.py     # Generates publication-quality figures
│   └── plots/              # Directory for saved figures
│
├── experiments/            # Driver scripts for specific experiments
│   ├── exp_01_trainability.py  # Exp 1: Depth Scaling (Barren Plateau analysis)
│   ├── exp_02_robustness.py    # Exp 2: Noise Resilience (Shot budget sweep)
│   └── exp_03_qml_utility.py   # Exp 3: QML Utility (Make Moons classification)
│
├── results/                # Raw logs and JSON metrics (Auto-generated per run)
│
├── src/                    # Core logic modules
│   ├── algorithms.py       # Optimizer implementations (Adam, SPSA, QuGStep)
│   ├── problems.py         # Task definitions (MaxCut, MakeMoons)
│   └── utils.py            # Logging, Configuration, and I/O utilities
│
├── main_simulation.py      # Quick-start script (Runs a reduced version of all experiments)
└── requirements.txt        # Python dependencies
```

## Installation and Usage

### Prerequisites
It is recommended to run this code in an isolated Python environment (Virtualenv or Conda).

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a Full Replication**
    To execute the rigorous benchmarks presented in the paper, run the experiment drivers in the following order. Note that these scripts utilize multiprocessing to accelerate execution.
* **Experiment 1: Trainability vs. Depth**
    Compares optimizers on 8-qubit MaxCut circuits with depths ranging from  to .
    ```bash
    python experiments/exp_01_trainability.py
    ```

* **Experiment 2: Noise Robustness**
    Tests convergence stability across varying shot budgets () to simulate different noise environments.
    ```bash
    python experiments/exp_02_robustness.py
    ```

* **Experiment 3: Quantum Machine Learning**
    Trains a Variational Quantum Classifier on the "Make Moons" non-linear dataset to assess generalization capability.
    ```bash
    python experiments/exp_03_qml_utility.py
    ```


3. **Generate Plots**
    After the experiments conclude, the raw JSON metrics are saved in the `results/` directory. Use the analysis script to parse the latest data and generate the figures used in the manuscript.
    ```bash
    python analysis/plot_results.py
    ```
    The figures will be saved to `analysis/plots/`.

## Key Results

### 1. Restoration of Trainability

In deep circuits (), standard Adam optimization collapses as gradients vanish below the shot noise floor. QuGStep restores a monotonic descent trajectory by filtering this noise, effectively navigating the Barren Plateau.

### 2. Noise Resilience

In high-noise regimes (), QuGStep achieves a final energy approximately 15x lower than standard gradient descent. This demonstrates that spatial averaging (large stencil size) is significantly more sample-efficient than temporal averaging (stochastic perturbation) for NISQ optimization.

### 3. QML Generalization

On the "Make Moons" geometric classification task, QuGStep yields decision boundaries with lower variance and higher median test accuracy compared to SPSA, indicating superior stability for machine learning applications.

## Configuration

Hyperparameters for the experiments can be modified directly within the experiment scripts found in the `experiments/` directory. Look for the configuration block at the top of each file:

```python
# experiments/exp_01_trainability.py

DEPTHS = [2, 4, 6, 8, 10, 12]   # Circuit Depths to sweep
TRIALS = 30                     # Independent trials for statistical significance
SHOTS = 1000                    # Simulated hardware shot noise budget
```