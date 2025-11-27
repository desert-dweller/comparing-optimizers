import json
import glob
import os
import csv
import datetime

# --- CONFIG ---
LOG_DIR = "logs"
SUMMARY_DIR = "results"
os.makedirs(SUMMARY_DIR, exist_ok=True)

def get_latest_log():
    """Finds the most recent benchmark JSON file."""
    files = glob.glob(os.path.join(LOG_DIR, "benchmark_data_*.json"))
    if not files: raise FileNotFoundError("No logs found! Run main.py first.")
    return max(files, key=os.path.getctime)

def calculate_metrics(custom_res, ref_res):
    """Calculates Exact Difference and Iteration Difference."""
    c_cost = custom_res['final_cost']
    r_cost = ref_res['final_cost']
    c_iter = custom_res['iterations']
    r_iter = ref_res['iterations']

    # 1. Exact Cost Difference
    # Negative means Custom was lower (better if minimizing)
    exact_diff = c_cost - r_cost

    # 2. Iteration Difference
    # Negative means Custom was faster (fewer steps)
    iter_diff = c_iter - r_iter
    
    return exact_diff, iter_diff

def main():
    try:
        log_file = get_latest_log()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"📊 Analyzing Log: {os.path.basename(log_file)}\n")
    
    with open(log_file, 'r') as f:
        data = json.load(f)

    # Define the pairs to compare
    # Format: (Custom Name, Built-in Name)
    comparisons = [
        ('Custom Adam', 'PyTorch Adam'),
        ('Custom Newton', 'SciPy Newton'),
        ('Custom BFGS', 'SciPy BFGS'),
        ('Custom NM', 'SciPy NM')
    ]

    # Prepare Data for CSV
    csv_rows = []
    headers = ["Function", "Comparison", "Exact Diff", "Iter Diff"]
    csv_rows.append(headers)

    # --- PROCESSING LOOP ---
    for func_name, results in data.items():
        for custom_name, ref_name in comparisons:
            
            # Check if both exist in the log
            if custom_name in results and ref_name in results:
                custom_res = results[custom_name]
                ref_res = results[ref_name]
                
                exact, iters = calculate_metrics(custom_res, ref_res)
                
                # Format Names for cleaner display
                comp_name = f"{custom_name.replace('Custom ','')} vs {ref_name.split(' ')[0]}"
                
                # Add to CSV List
                csv_rows.append([func_name, comp_name, exact, iters])        

    # --- SAVE CSV ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = os.path.join(SUMMARY_DIR, f"comparison_analysis_{timestamp}.csv")
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        
    print(f"\n✅ Comparison saved to: {csv_filename}")

if __name__ == "__main__":
    main()