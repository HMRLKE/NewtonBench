import os
import json
import pandas as pd
from typing import List, Dict, Any

def aggregate_results(base_dir: str = "evaluation_results") -> pd.DataFrame:
    all_results = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return pd.DataFrame()

    for root, dirs, files in os.walk(base_dir):
        if "agg_results.json" in files:
            file_path = os.path.join(root, "agg_results.json")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Local path handling
                rel_path = os.path.relpath(root, base_dir)
                parts = rel_path.split(os.sep)
                
                # Expected: {model}/{module}/{agent}/{difficulty}/{version}/{expt_name}
                if len(parts) >= 6:
                    model = parts[0]
                    module = parts[1]
                    agent = parts[2]
                    difficulty = parts[3]
                    version = parts[4]
                    expt_name = parts[5]
                else:
                    model = module = agent = difficulty = version = expt_name = "unknown"

                # Parse expt_name for noise, prompt, consistency if possible
                noise = "unknown"
                prompt = "unknown"
                consistency = "unknown"
                
                if "noise" in expt_name:
                    noise = expt_name.split("noise")[1].split("_")[0]
                if "prompt" in expt_name:
                    prompt = expt_name.split("prompt_")[1].split("_")[0]
                if "consistent" in expt_name:
                    consistency = "True" if "inconsistent" not in expt_name else "False"

                res_entry = {
                    "Model": model,
                    "Module": module,
                    "Agent": agent,
                    "Difficulty": difficulty,
                    "Version": version,
                    "Noise": noise,
                    "PromptSet": prompt,
                    "Consistency": consistency,
                    "Exact_Acc": data.get("exact_accuracy", 0.0),
                    "Mean_RMSLE": data.get("mean_rmsle", float('nan')),
                    "Symbolic_Equiv": data.get("symbolic_equivalent_rate", 0.0),
                    "Success_Rate": data.get("success_rate", 0.0),
                    "Total_Trials": data.get("total_trials", 0)
                }
                all_results.append(res_entry)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    df = aggregate_results()
    if not df.empty:
        # Sort for better view
        df = df.sort_values(by=["Module", "Version", "Noise", "PromptSet", "Consistency"])
        
        # Save to CSV
        df.to_csv("benchmark_summary.csv", index=False)
        print("Report saved to benchmark_summary.csv")
        
        # Print a nice Markdown table
        print("\n### Benchmark Summary Table\n")
        print(df.to_markdown(index=False))
    else:
        print("No results found to aggregate.")
