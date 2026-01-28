import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PLOTTING STYLE ---
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams.update({
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {'Adam': '#1f77b4', 'SPSA': '#ff7f0e', 'QuGStep': '#d62728'}
MARKERS = {'Adam': 'o', 'SPSA': 's', 'QuGStep': '^'}

def load_latest_experiment(exp_prefix, results_dir="results"):
    """Finds the most recent folder starting with exp_prefix."""
    search_path = os.path.join(results_dir, f"{exp_prefix}*")
    folders = glob.glob(search_path)
    if not folders:
        print(f"⚠️ No results found for {exp_prefix}")
        return None
    
    # Sort by creation time (newest first)
    latest_folder = max(folders, key=os.path.getmtime)
    json_path = os.path.join(latest_folder, "metrics.json")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded: {json_path}")
        return pd.DataFrame(data['results'])
    except FileNotFoundError:
        print(f"❌ Metrics file missing in {latest_folder}")
        return None

def plot_trainability(df):
    """Figure 1: Depth Scaling (Barren Plateau)."""
    if df is None: return
    
    # Group by Depth and Optimizer
    summary = df.groupby(['depth', 'opt'])['final_cost'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(8, 6))
    for opt in COLORS:
        subset = summary[summary['opt'] == opt]
        if subset.empty: continue
        
        plt.errorbar(
            subset['depth'], subset['mean'], yerr=subset['std'],
            label=opt, color=COLORS[opt], marker=MARKERS[opt], capsize=5
        )
        
    plt.xlabel("Circuit Depth (L)")
    plt.ylabel("Final Energy (H)")
    plt.title("Trainability Limit (MaxCut)")
    plt.legend()
    plt.savefig("analysis/fig1_trainability.png", dpi=300)
    plt.close()

def plot_robustness(df):
    """Figure 2: Noise Resilience."""
    if df is None: return
    
    summary = df.groupby(['shots', 'opt'])['final_cost'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(8, 6))
    for opt in COLORS:
        subset = summary[summary['opt'] == opt]
        if subset.empty: continue
        
        plt.errorbar(
            subset['shots'], subset['mean'], yerr=subset['std'],
            label=opt, color=COLORS[opt], marker=MARKERS[opt], capsize=5
        )
    
    plt.xscale('log')
    plt.xlabel("Shot Budget (Inverse Noise)")
    plt.ylabel("Final Energy")
    plt.title("Noise Resilience (Depth 8)")
    plt.legend()
    plt.savefig("analysis/fig2_robustness.png", dpi=300)
    plt.close()

def plot_qml_utility(df):
    """Figure 3: QML Accuracy."""
    if df is None: return
    
    # Boxplot is better for accuracy distribution
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='opt', y='accuracy_score', data=df, palette=COLORS)
    
    plt.ylabel("Test Accuracy")
    plt.title("QML Generalization (Make Moons)")
    plt.savefig("analysis/fig3_qml_utility.png", dpi=300)
    plt.close()

def main():
    os.makedirs("analysis", exist_ok=True)
    
    print("--- GENERATING PLOTS ---")
    
    # 1. Trainability
    df_train = load_latest_experiment("Exp01")
    plot_trainability(df_train)
    
    # 2. Robustness
    df_robust = load_latest_experiment("Exp02")
    plot_robustness(df_robust)
    
    # 3. QML
    df_qml = load_latest_experiment("Exp03")
    plot_qml_utility(df_qml)
    
    print("✨ All plots saved to analysis/ folder.")

if __name__ == "__main__":
    main()