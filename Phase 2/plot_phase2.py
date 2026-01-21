import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

RESULTS_DIR = "results_phase2"
PLOTS_DIR = "plots_phase2"
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_latest_result():
    files = glob.glob(os.path.join(RESULTS_DIR, "FINAL_BENCHMARK_*.json"))
    if not files: raise FileNotFoundError("No results found!")
    return max(files, key=os.path.getctime)

def plot_benchmark(data, filename_base):
    # 1. SCALING PLOT (Cost vs Depth)
    plt.figure(figsize=(10, 6))
    depths = sorted([int(k.split('_')[1]) for k in data.keys()])
    
    # Define styles for categories
    styles = {
        'SPSA': ('gray', '--', 'o'),
        'Adam': ('tab:blue', '-', 's'),
        'BFGS': ('tab:green', '-', '^'),
        'Hybrid': ('tab:red', '-', '*'),
        'Saddle': ('tab:purple', '-', 'D'),
        'Nelder': ('tab:orange', ':', 'x')
    }

    # Extract all optimizer names from the first depth
    first_depth = f"Depth_{depths[0]}"
    opt_names = data[first_depth].keys()

    for opt in opt_names:
        costs = []
        for d in depths:
            costs.append(data[f"Depth_{d}"][opt]['final_cost'])
        
        # Determine Style
        c, ls, m = 'black', '-', '.'
        if 'SPSA' in opt: c, ls, m = styles['SPSA']
        elif 'Hybrid' in opt: c, ls, m = styles['Hybrid'] # Highlight Winner
        elif 'Saddle' in opt: c, ls, m = styles['Saddle']
        elif 'Adam' in opt: c, ls, m = styles['Adam']
        elif 'BFGS' in opt: c, ls, m = styles['BFGS']
        elif 'Nelder' in opt: c, ls, m = styles['Nelder']
        
        # Plot
        plt.plot(depths, costs, label=opt.split('_')[1], color=c, linestyle=ls, marker=m, linewidth=2)

    plt.title("Scaler: Noise Resilience vs Circuit Depth")
    plt.xlabel("Circuit Depth")
    plt.ylabel("Final Cost (Lower is Better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"scaling_{filename_base}.png"), dpi=150)
    print("Saved Scaling Plot.")

if __name__ == "__main__":
    f = get_latest_result()
    with open(f, 'r') as file: data = json.load(file)
    plot_benchmark(data, os.path.basename(f).replace(".json", ""))