import argparse
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import hashlib

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_scientist import adapter, sr, kg, reviewer, causal_env, causal_discovery

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_physics_mode(args, run_dir, config):
    # 0. Resolve Physics Config (Difficulty & Law Version)
    # We need to resolve these here to ensure we know exactly what is run
    try:
        law_module = importlib.import_module(f"modules.{args.task}.laws")
        get_available_law_versions = law_module.get_available_law_versions
    except:
        # Fallback if module doesn't have laws.py or it's non-standard
        def get_available_law_versions(diff): return ['v0', 'v1', 'v2']
    
    difficulty = getattr(args, 'difficulty', 'easy') # Default if not passed
    law_version = getattr(args, 'law_version', None)
    
    # If law_version is None, pick one to make it deterministic for this run
    if law_version is None:
        import random
        # Note: This simple random choice might duplicate logic in get_ground_truth_law if not careful,
        # but to save it, we must decide it now.
        # Ideally, we should refactor get_ground_truth_law to accept a "resolved" version, 
        # but for now we will trust get_ground_truth_law handles 'None' by randomizing.
        # WAIT: if we let get_ground_truth_law randomize, we won't know which one it picked unless we modify it 
        # or we pick it here.
        # Let's pick it here.
        available = get_available_law_versions(difficulty)
        # Re-seed random just in case to ensure args.seed controls this
        random.seed(args.seed)
        law_version = random.choice(available)
        
    print(f"Physics Config: Difficulty={difficulty}, Law Version={law_version}")
    
    # Update config with resolved values
    config['difficulty'] = difficulty
    config['law_version'] = law_version
    
    # Update config.json with new details
    from mini_scientist import visualize
    gt_eqn_str = visualize.get_gt_equation_string(difficulty, law_version, task_name=args.task)
    config['gt_eqn_str'] = gt_eqn_str
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 1. Adapter: Collect Data
    print("Step 1: collecting data...")
    try:
        # We need to pass difficulty and law_version to adapter? 
        # adapter.get_data currently doesn't accept them, it relies on module internals 
        # or defaults. 
        # PoC Limitation: adapter.get_data is generic. 
        # However, run_experiment_for_module in module handles it.
        # We need to pass these to the experiment runner called INSIDE adapter?
        # Actually adapter.get_data for PoC calls module.run_experiment_for_module.
        # We need to update adapter.get_data to accept these or monkey patch.
        # Let's check adapter.get_data signature.
        # It DOES NOT accept difficulty/law_version.
        # REQUIRED CHANGE: Update adapter.get_data to accept generic kwargs to pass down.
        
        # For now, let's assume we update adapter.get_data or pass it via kwargs if we change it.
        # Let's modify adapter.get_data first/simultaneously.
        df = adapter.generate_data(args.task, args.n_samples, args.noise, 
                                   difficulty=difficulty, law_version=law_version)
    except ValueError as e:
        print(f"Error: {e}")
        return []
    except TypeError as e:
         print(f"Error calling adapter: {e}. defaulting to old behavior")
         df = adapter.generate_data(args.task, args.n_samples, args.noise)


    # 2. Split Data
    print("Step 2: splitting data...")
    train_df, holdout_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    
    train_df.to_csv(os.path.join(run_dir, "dataset_train.csv"), index=False)
    holdout_df.to_csv(os.path.join(run_dir, "dataset_holdout.csv"), index=False)
    
    # 3. Symbolic Regression
    print("Step 3: running symbolic regression (this may take a moment)...")
    if 'target' not in train_df.columns:
        print("Error: 'target' column not found in data.")
        return []
        
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = holdout_df.drop(columns=['target'])
    y_test = holdout_df['target']
    
    n_iterations = getattr(args, 'n_iterations', 1000)
    parsimony = getattr(args, 'parsimony', 0.001)
    
    sr_result = sr.run_sr(X_train, y_train, n_iterations=n_iterations, parsimony=parsimony, temp_dir=os.path.join(run_dir, "pysr_tmp"))
    
    # Serialize results safely
    serializable_result = {}
    for k, v in sr_result.items():
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            serializable_result[k] = v
        else:
            serializable_result[k] = str(v)

    with open(os.path.join(run_dir, "sr_result.json"), "w") as f:
        json.dump(serializable_result, f, indent=2)
        
    with open(os.path.join(run_dir, "equation.txt"), "w") as f:
        f.write(sr_result['equation'])

    print(f"Discovered Equation: {sr_result['equation']}")

    # Evaluation on Holdout
    import sympy
    import numpy as np
    
    expr = sr_result['sympy_expr']
    try:
        f_eval = sympy.lambdify([sympy.Symbol(c) for c in X_test.columns], expr, modules=["numpy"])
        y_pred = f_eval(**{c: X_test[c].values for c in X_test.columns})
        
        def calculate_rmsle(yt, yp):
            yt, yp = np.abs(np.array(yt)), np.abs(np.array(yp))
            return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt))**2)))

        # Also check symbolic equivalence with SymPy
        # AI Newton Standard: ignore specific values of constants
        symbolic_match = False
        try:
            gt_eqn_str = config.get('gt_eqn_str', '')
            if gt_eqn_str:
                gt_expr = sympy.sympify(gt_eqn_str, locals=kg.SYMPY_LOCALS)
                
                # Helper to replace all numbers with a symbol 'C'
                def ignore_constants(e):
                    for n in e.atoms(sympy.Number):
                        e = e.subs(n, sympy.Symbol('C'))
                    return e

                # Check if structural form is identical
                diff = sympy.simplify(ignore_constants(expr) - ignore_constants(gt_expr))
                symbolic_match = (diff == 0)
        except:
            symbolic_match = False

        metrics = {
            "r2": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmsle": calculate_rmsle(y_test, y_pred),
            "symbolic_match": symbolic_match
        }
    except Exception as e:
        print(f"Warning: could not evaluate equation on holdout: {e}")
        metrics = {"error": str(e)}

    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 4. Equation Graph KG
    print("Step 4: building knowledge graph...")
    kg_data = kg.equation_to_kg(sr_result['equation'])
    with open(os.path.join(run_dir, "kg.json"), "w") as f:
        json.dump(kg_data, f, indent=2)

    # 5. Automated Reviewer
    print("Step 5: generating paper...")
    reviewer.generate_paper(run_dir, config, sr_result, kg_data, metrics)
    
    return [
        "config.json", "dataset_train.csv", "dataset_holdout.csv",
        "sr_result.json", "equation.txt", "experiment.json",
        "kg.json", "paper.md"
    ]

def run_causal_mode(args, run_dir, config):
    print("Step 1: initializing causal environment...")
    # n_nodes is hardcoded for PoC but could be arg
    env = causal_env.CausalEnv(n_nodes=10, seed=args.seed)
    
    # Save Ground Truth
    gt_dag = env.get_ground_truth()
    import networkx as nx
    gt_json = nx.node_link_data(gt_dag)
    with open(os.path.join(run_dir, "ground_truth.json"), "w") as f:
        json.dump(gt_json, f, indent=2)

    print("Step 2: running causal discovery...")
    discoverer = causal_discovery.CausalDiscoverer(env)
    pred_adj = discoverer.discover()
    
    # Save Intervention Log
    with open(os.path.join(run_dir, "intervention_log.json"), "w") as f:
        json.dump(discoverer.history, f, indent=2)
        
    print("Step 3: evaluating results...")
    # Convert prediction to NetworkX for comparison
    pred_G = nx.DiGraph()
    pred_G.add_nodes_from(env.nodes)
    for u, targets in pred_adj.items():
        for v in targets:
            pred_G.add_edge(u, v)

    # Calculate metrics (SHD, F1)
    tp = len(set(gt_dag.edges()) & set(pred_G.edges()))
    fp = len(set(pred_G.edges()) - set(gt_dag.edges()))
    fn = len(set(gt_dag.edges()) - set(pred_G.edges()))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "true_edges": len(gt_dag.edges()),
        "pred_edges": len(pred_G.edges()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Log simplified metrics
    print(f"Graph Metrics: F1={f1:.2f} (Prec={precision:.2f}, Rec={recall:.2f})")
    
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 4. Result Graph
    kg_data = causal_discovery.graph_to_json(pred_adj)
    with open(os.path.join(run_dir, "kg.json"), "w") as f:
        json.dump(kg_data, f, indent=2)

    # 5. Automated Reviewer
    print("Step 4: generating paper...")
    # Pass 'causal_result' instead of sr_result
    causal_result = {"type": "causal_graph", "nodes": env.nodes, "metrics": metrics}
    reviewer.generate_paper(run_dir, config, causal_result, kg_data, metrics)

    return [
        "config.json", "ground_truth.json", "intervention_log.json",
        "experiment.json", "kg.json", "paper.md"
    ]

def main():
    parser = argparse.ArgumentParser(description="Mini AI-Scientist Loop")
    parser.add_argument("--mode", type=str, default="physics", choices=["physics", "causal"], help="Operation mode")
    parser.add_argument("--task", type=str, default="m0_gravity", help="Task name (physics) or config (causal)")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level")
    parser.add_argument("--n_iterations", type=int, default=1000, help="SR iterations")
    parser.add_argument("--parsimony", type=float, default=0.001, help="Complexity penalty")
    parser.add_argument("--law_version", type=str, default=None, help="Specific law version")
    parser.add_argument("--output_dir", type=str, default="runs", help="Base output directory")
    
    args = parser.parse_args()
    
    if args.task == "list" and args.mode == "physics":
        print("Available physics tasks:")
        for t in adapter.get_available_tasks():
            print(f" - {t}")
        return

    print(f"--- Starting Mini AI-Scientist Run: {args.task} (Mode: {args.mode}) ---")
    
    # Setup Run Directory
    run_id = f"{args.mode}_{args.task}_{args.seed}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save Config
    config = vars(args)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    if args.mode == "physics":
        artifacts = run_physics_mode(args, run_dir, config)
    else:
        artifacts = run_causal_mode(args, run_dir, config)
    
    # 6. Manifest
    print("Creating artifact manifest...")
    manifest = {}
    if artifacts:
        for art in artifacts:
            path = os.path.join(run_dir, art)
            if os.path.exists(path):
                manifest[art] = calculate_sha256(path)
            
    with open(os.path.join(run_dir, "artifacts_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n--- Run Complete. Output in {run_dir} ---")

if __name__ == "__main__":
    main()
