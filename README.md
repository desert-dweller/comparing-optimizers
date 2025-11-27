# Benchmarking Classical Optimization Algorithms for Non-Convex Landscapes

**Author:** Karim El Tohamy  
**Course:** Scientific Computing (CIT644)  

## 📌 Overview

This project implements a comprehensive library of numerical optimization algorithms from first principles and benchmarks them against industry-standard implementations (`SciPy` and `PyTorch`). 

The core innovation is a **"Black Box" Automatic Numerical Differentiation Engine** that calculates gradients and Hessians using Central Finite Difference, removing the need for manual derivative formulas.

### Algorithms Implemented (From Scratch)
1.  **Adam:** Adaptive Moment Estimation (First-Order).
2.  **Newton's Method:** Exact Hessian Inversion (Second-Order).
3.  **BFGS:** Quasi-Newton Approximation.
4.  **Nelder-Mead:** Gradient-Free Simplex Method.

---

## 📂 Project Structure

```text
├── optimizers.py       # Core library containing the Optimizer classes and Auto-Diff engine
├── benchmarks.py       # Suite of 7 pathological test functions (Rosenbrock, Himmelblau, etc.)
├── main.py             # Main runner: executes experiments, logs data to JSON, saves CSV summaries
├── plot_results.py     # Visualization: generates Convergence and Trajectory plots from logs
├── compare_results.py  # Analysis: calculates exact error metrics vs. SciPy/PyTorch
├── logs/               # (Generated) Raw JSON logs of optimization history
├── plots/              # (Generated) High-res PNG plots
└── results/            # (Generated) Summary CSV files