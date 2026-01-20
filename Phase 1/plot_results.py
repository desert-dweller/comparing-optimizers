import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import glob
import os
from benchmarks import test_cases

# --- CONFIG ---
LOG_DIR = "logs"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def get_latest_log():
    files = glob.glob(os.path.join(LOG_DIR, "benchmark_data_*.json"))
    if not files: raise FileNotFoundError("No logs found!")
    return max(files, key=os.path.getctime)

# --- STYLE HELPER ---
def get_style(opt_name):
    """Returns (linestyle, alpha, color, marker_symbol) based on optimizer name."""
    ls = '-'
    alpha = 0.8
    color = 'black'
    marker = 'o' # Default
    
    # 1. Base Colors & Markers
    if 'Adam' in opt_name:   
        color = 'tab:cyan'
        marker = '*'
    if 'Newton' in opt_name: 
        color = 'tab:orange'
        marker = 'o'
    if 'BFGS' in opt_name:   
        color = 'tab:green'
        marker = '^'
    if 'NM' in opt_name:     
        color = 'tab:red'
        marker = 's'
    
    # 2. SciPy / PyTorch Modifiers
    if 'SciPy' in opt_name or 'PyTorch' in opt_name:
        ls = '--'
        alpha = 0.6
        # Darken color slightly for contrast
        try:
            c = mcolors.to_rgb(color)
            # Ensure a fixed 3-tuple (r, g, b)
            r, g, b = c
            color = mcolors.to_hex((max(0.0, r * 0.8), max(0.0, g * 0.8), max(0.0, b * 0.8)))
        except Exception:
            pass  # Fallback if color is not RGB
        
    return ls, alpha, color, marker

# --- PLOT 1: CONVERGENCE ---
def plot_convergence(func_name, results):
    plt.figure(figsize=(10, 6))
    for opt_name, data in results.items():
        costs = [step['cost'] for step in data['history']]
        ls, alpha, color, _ = get_style(opt_name)
        
        # Thinner lines for convergence plot too
        plt.semilogy(costs, label=opt_name, linestyle=ls, linewidth=1.5, alpha=alpha, color=color)

    plt.title(f'Convergence: {func_name}', fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{func_name}_convergence.png"), dpi=150)
    plt.close()

# --- PLOT 2: TRAJECTORY ---
def plot_trajectory(func_name, results, cost_func_obj):
    if len(cost_func_obj.start_params) != 2: return

    plt.figure(figsize=(10, 8))
    
    # 1. Zoom Box
    if func_name == "Rosenbrock": x_lim, y_lim = (-1.5, 1.5), (-0.5, 1.5)
    elif func_name == "Himmelblau": x_lim, y_lim = (-5.0, 5.0), (-5.0, 5.0)
    elif func_name == "Ellipse": x_lim, y_lim = (-5.0, 5.0), (-5.0, 5.0)
    elif func_name == "Booth": x_lim, y_lim = (-7.0, 7.0), (-2.0, 14.0)
    elif func_name == "Matyas": x_lim, y_lim = (-1.0, 6.0), (-1.0, 6.0)
    elif func_name == "ThreeHump": x_lim, y_lim = (-2.5, 2.5), (-2.5, 2.5)
    else: x_lim, y_lim = (-3.5, 3.5), (-3.5, 3.5)
    
    # 2. Background
    x_range = np.linspace(x_lim[0], x_lim[1], 100)
    y_range = np.linspace(y_lim[0], y_lim[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = cost_func_obj.func([X[i,j], Y[i,j]])
            
    plt.contourf(X, Y, np.log(Z + 1e-9), levels=30, cmap='gray_r', alpha=0.15)
    plt.colorbar(label='Log(Cost)')

    # 3. Trajectories
    for opt_name, data in results.items():
        # --- FILTER: SKIP CUSTOM BFGS ---
        if opt_name == "Custom BFGS": continue
        # --------------------------------

        path = np.array([step['params'] for step in data['history']])
        if len(path) == 0: continue

        # Get Matched Styles
        ls, alpha, color, marker_symbol = get_style(opt_name)
        
        # Plot Line: Thinner (1.5) and more transparent (alpha * 0.7)
        plt.plot(path[:, 0], path[:, 1], label=opt_name, 
                 linestyle=ls, linewidth=1.5, alpha=alpha * 0.7, color=color, marker=None)
        
        # Start Point: Smaller size (8)
        if x_lim[0] < path[0,0] < x_lim[1] and y_lim[0] < path[0,1] < y_lim[1]:
            plt.plot(path[0,0], path[0,1], marker=marker_symbol, color=color, 
                     markersize=8, zorder=10, label='_nolegend_') 
        
        # End Point: Smaller size (6)
        if x_lim[0] < path[-1,0] < x_lim[1] and y_lim[0] < path[-1,1] < y_lim[1]:
            plt.plot(path[-1,0], path[-1,1], 'rx', markersize=6, zorder=10, label='_nolegend_')

    plt.title(f'Trajectory: {func_name}', fontsize=14)
    plt.xlabel('Parameter 0')
    plt.ylabel('Parameter 1')
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(PLOT_DIR, f"{func_name}_trajectory_zoomed.png")
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    try:
        log_file = get_latest_log()
        print(f"Visualizing log: {log_file}")
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        for case in test_cases:
            if case.name in data:
                plot_convergence(case.name, data[case.name])
                plot_trajectory(case.name, data[case.name], case)
    except FileNotFoundError:
        print("Log file not found. Please run main.py first!")