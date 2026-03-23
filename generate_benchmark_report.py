import os
import json
import re
import pandas as pd
from typing import List, Dict, Any

def aggregate_results(base_dir: str = "evaluation_results") -> pd.DataFrame:
    all_results = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return pd.DataFrame()

    for root, dirs, files in os.walk(base_dir):
        if "aggregated_results.json" in files:
            file_path = os.path.join(root, "aggregated_results.json")
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
                noise = "0.0"
                prompt = "original"
                consistency = "False"
                
                # More robust parsing for noise0_0 (0.0) or noise0_1 (0.1)
                noise_match = re.search(r'noise(\d+)_(\d+)', expt_name)
                if noise_match:
                    noise = f"{noise_match.group(1)}.{noise_match.group(2)}"
                elif "noise" in expt_name:
                    noise = expt_name.split("noise")[1].split("_")[0]

                if "prompt_modified" in expt_name:
                    prompt = "modified"
                elif "prompt_original" in expt_name:
                    prompt = "original"
                
                if "consistent" in expt_name:
                    consistency = "True" if "inconsistent" not in expt_name else "False"

                # Navigate the nested JSON structure
                agg = data.get("aggregate", {})
                all_trials = agg.get("all_trials", {})
                
                res_entry = {
                    "Model": model,
                    "Module": module,
                    "Agent": agent,
                    "Difficulty": difficulty,
                    "Version": version,
                    "Noise": noise,
                    "PromptSet": prompt,
                    "Consistency": consistency,
                    "Exact_Acc": all_trials.get("average_exact_accuracy", data.get("exact_accuracy", 0.0)),
                    "Mean_RMSLE": all_trials.get("average_rmsle", data.get("mean_rmsle", float('nan'))),
                    "Success_Rate": all_trials.get("success_rate", data.get("success_rate", 0.0)),
                    "Total_Trials": all_trials.get("num_total_trials", data.get("total_trials", 0))
                }
                all_results.append(res_entry)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    
    # Aggregate results for the same configuration
    # (sometimes multiple folders exist for the same logical experiment)
    group_cols = ["Model", "Module", "Agent", "Difficulty", "Version", "Noise", "PromptSet", "Consistency"]
    
    # Convert numerical columns to float for proper mean calculation
    df["Exact_Acc"] = pd.to_numeric(df["Exact_Acc"], errors='coerce')
    df["Mean_RMSLE"] = pd.to_numeric(df["Mean_RMSLE"], errors='coerce')
    df["Success_Rate"] = pd.to_numeric(df["Success_Rate"], errors='coerce')
    df["Total_Trials"] = pd.to_numeric(df["Total_Trials"], errors='coerce')

    agg_df = df.groupby(group_cols).agg({
        "Exact_Acc": "mean",
        "Mean_RMSLE": "mean",
        "Success_Rate": "mean",
        "Total_Trials": "sum"
    }).reset_index()

    return agg_df

if __name__ == "__main__":
    df = aggregate_results()
    if not df.empty:
        # Sort for better view
        df = df.sort_values(by=["Model", "Module", "Difficulty", "Version", "Noise", "PromptSet", "Consistency"])
        
        # Format percentages and decimals for better readability
        df["Exact_Acc"] = df["Exact_Acc"].apply(lambda x: f"{x*100:>.1f}%")
        df["Success_Rate"] = df["Success_Rate"].apply(lambda x: f"{x*100:>.1f}%")
        df["Mean_RMSLE"] = df["Mean_RMSLE"].apply(lambda x: f"{x:>.4f}" if pd.notnull(x) else "NaN")
        
        # Save to CSV
        df.to_csv("benchmark_summary.csv", index=False)
        print("Report saved to benchmark_summary.csv")
        
        # Print a nice Markdown table
        print("\n### Benchmark Summary Table\n")
        try:
            # Try to use tabulate for pretty printing
            from tabulate import tabulate
            print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
        except ImportError:
            # Fallback to pandas string representation
            print(df.to_string(index=False))
    else:
        print("No results found to aggregate.")
