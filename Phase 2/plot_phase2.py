import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# --- CONFIG ---
RESULTS_DIR = "results_phase2"
PLOTS_DIR = "plots_phase2"
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_latest_result():
    """Finds the most recent Phase 2 result file."""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    if not files:
        raise FileNotFoundError("No results found in results_phase2/")
    return max(files, key=os.path.getctime)

def plot_barren_plateau_scaling(data, filename_base):
    """
    Plot 1: The 'Survival Curve'
    X-Axis: Circuit Depth (Layers)
    Y-Axis: Final Cost (Lower is Better)
    """
    depths = []
    std_costs = []
    qug_costs = []
    
    # Sort keys to ensure plotting order (Depth_2, Depth_5, etc.)
    sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    
    for key in sorted_keys:
        depth = int(key.split('_')[1])
        res = data[key]
        
        depths.append(depth)
        std_costs.append(res['Standard_Adam']['final_cost'])
        qug_costs.append(res['QuGStep_Adam']['final_cost'])
        
    plt.figure(figsize=(10, 6))
    
    # Plot Standard Adam
    plt.plot(depths, std_costs, 'o--', label='Standard Adam (Fixed $\epsilon$)', 
             color='gray', linewidth=2, markersize=8)
    
    # Plot QuGStep Adam
    plt.plot(depths, qug_costs, '*-', label='QuGStep Adam (Adaptive $\epsilon$)', 
             color='tab:red', linewidth=3, markersize=10)
    
    plt.title("The 'Barren Plateau' Survival Test", fontsize=14)
    plt.xlabel("Circuit Depth (Layers)", fontsize=12)
    plt.ylabel("Final Cost (1 - Fidelity)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Annotate the "Win"
    plt.annotate('Noise Floor / Plateau', xy=(depths[-1], std_costs[-1]), 
                 xytext=(depths[-1]-2, std_costs[-1]+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    save_path = os.path.join(PLOTS_DIR, f"scaling_{filename_base}.png")
    plt.savefig(save_path, dpi=150)
    print(f"   Saved Scaling Plot: {save_path}")
    plt.close()

def plot_convergence_comparison(data, filename_base):
    """
    Plot 2: Convergence Comparison for the Deepest Circuit
    X-Axis: Iterations
    Y-Axis: Cost
    """
    # Find the deepest depth run
    deepest_key = max(data.keys(), key=lambda x: int(x.split('_')[1]))
    depth = deepest_key.split('_')[1]
    
    std_hist = data[deepest_key]['Standard_Adam']['history']
    qug_hist = data[deepest_key]['QuGStep_Adam']['history']
    
    std_costs = [x['cost'] for x in std_hist]
    qug_costs = [x['cost'] for x in qug_hist]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(std_costs, label=f'Standard Adam (Depth {depth})', 
             color='gray', alpha=0.6, linewidth=1.5)
    plt.plot(qug_costs, label=f'QuGStep Adam (Depth {depth})', 
             color='tab:red', linewidth=2.5)
    
    plt.title(f"Convergence at Depth {depth} (High Noise Regime)", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.yscale('log') # Log scale helps see fine convergence
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    save_path = os.path.join(PLOTS_DIR, f"convergence_depth{depth}_{filename_base}.png")
    plt.savefig(save_path, dpi=150)
    print(f"   Saved Convergence Plot: {save_path}")
    plt.close()

if __name__ == "__main__":
    try:
        json_file = get_latest_result()
        print(f"📊 Analyzing: {os.path.basename(json_file)}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        base_name = os.path.basename(json_file).replace(".json", "")
        
        plot_barren_plateau_scaling(data, base_name)
        plot_convergence_comparison(data, base_name)
        
        print("\n✅ Visualization Complete. Check the 'plots_phase2' folder.")
        
    except Exception as e:
        print(f"Error: {e}")