import os
import json
import time
import argparse
import sys
import shutil
from typing import List

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_scientist import adapter, run, kg, visualize

def update_global_kg(run_dir, accumulation_dir):
    """
    Reads the KG from the run directory and merges/adds it to the global KG.
    """
    kg_path = os.path.join(run_dir, "kg.json")
    if not os.path.exists(kg_path):
        print(f"Warning: No KG found in {run_dir}")
        return

    with open(kg_path, "r") as f:
        new_kg = json.load(f)
        
    global_kg_path = os.path.join(accumulation_dir, "global_kg.json")
    
    # Initialize Structure
    if os.path.exists(global_kg_path):
        with open(global_kg_path, "r") as f:
            global_kg = json.load(f)
    else:
        global_kg = {
            "laws": [], 
            "graph": {"nodes": [], "edges": []},
            "gt_graph": {"nodes": [], "edges": []} # Added Ground Truth Graph
        }
        
    # Get run config for context
    config_path = os.path.join(run_dir, "config.json")
    sr_path = os.path.join(run_dir, "sr_result.json")
    
    task_name = "unknown"
    difficulty = "easy"
    law_version = "v0"
    sr_loss = None
    
    if os.path.exists(config_path):
         with open(config_path, 'r') as f:
             conf = json.load(f)
             task_name = conf.get('task', 'unknown')
             difficulty = conf.get('difficulty', 'easy')
             law_version = conf.get('law_version', 'v0')
             
    metrics = {}
    exp_path = os.path.join(run_dir, "experiment.json")
    if os.path.exists(exp_path):
        with open(exp_path, 'r') as f:
            metrics = json.load(f)

    if os.path.exists(sr_path):
        with open(sr_path, 'r') as f:
            sr_info = json.load(f)
            # Use SR loss if experiment.json doesn't have it
            if 'loss' not in metrics:
                metrics['loss'] = sr_info.get('loss')

    # --- Ground Truth Handling ---
    try:
        gt_eqn_str = visualize.get_gt_equation_string(difficulty, law_version, task_name=task_name)
        gt_kg = kg.equation_to_kg(gt_eqn_str)
    except Exception as e:
        print(f"Error generating GT for {task_name}: {e}")
        gt_eqn_str = "Error"
        gt_kg = {"nodes": [], "links": []}

    
    # Extract discovered equation info
    equation = new_kg.get('graph', {}).get('equation', 'unknown')

    # Calculate KG Similarity Score
    similarity_score = 0.0
    try:
        print(f"DEBUG Similarity: Comparing Disc='{equation}' vs GT='{gt_eqn_str}'")
        similarity_score = kg.calculate_kg_similarity(new_kg, gt_kg)
        print(f"DEBUG Similarity Score: {similarity_score:.4f}")
    except Exception as e:
        print(f"Error calculating similarity: {e}")

    # Add to laws list (Check if exists)
    exists = False
    for l in global_kg["laws"]:
        if l["task"] == task_name: 
            l["equation"] = equation
            l["gt_equation"] = gt_eqn_str
            l["similarity"] = similarity_score
            l["loss"] = float(metrics.get("loss", 0))
            l["rmsle"] = float(metrics.get("rmsle", 0))
            l["symbolic_match"] = bool(metrics.get("symbolic_match", False))
            l["timestamp"] = time.time()
            exists = True
            break
            
    if not exists:
        global_kg["laws"].append({
            "task": task_name,
            "equation": equation,
            "gt_equation": gt_eqn_str,
            "similarity": similarity_score,
            "loss": float(metrics.get("loss", 0)),
            "rmsle": float(metrics.get("rmsle", 0)),
            "symbolic_match": bool(metrics.get("symbolic_match", False)),
            "timestamp": time.time()
        })
        
    # --- Merge Graphs helper ---
    def merge_into_graph(target_graph_key, source_kg):
        current_nodes = global_kg[target_graph_key]["nodes"]
        current_edges = global_kg[target_graph_key]["edges"]
        
        # 1. Build Lookup for existing distinctive nodes (variables, numbers)
        # Operators are structural (AST) and should NOT be merged effectively unless we do sub-graph isomorphism which is hard.
        # We merge variables/constants to show "shared concepts".
        node_lookup = {}
        for n in current_nodes:
            if n['type'] in ['variable', 'number']:
                key = (n['label'], n['type'])
                # Only keep the FIRST one found if duplicates exist already
                if key not in node_lookup:
                    node_lookup[key] = n['id']
        
        current_max_id = 0
        if current_nodes:
            current_max_id = max(n['id'] for n in current_nodes) + 1
            
        # Map old IDs to new IDs
        id_map = {}
        
        # 2. Process Nodes
        for node in source_kg['nodes']:
            old_id = node['id']
            n_type = node.get('type', 'default')
            n_label = str(node.get('label', node.get('id')))
            
            # Check for existing
            key = (n_label, n_type)
            
            if n_type in ['variable', 'number'] and key in node_lookup:
                # Reuse existing node
                id_map[old_id] = node_lookup[key]
            else:
                # Create NEW node
                new_id = current_max_id
                current_max_id += 1
                id_map[old_id] = new_id
                
                n_copy = node.copy()
                n_copy['id'] = new_id
                n_copy['task'] = task_name # Tag with origin task
                
                current_nodes.append(n_copy)
                
                # Add to lookup if it's a reuseable type
                if n_type in ['variable', 'number']:
                    node_lookup[key] = new_id
            
        # 3. Process Edges
        links_key = 'links' if 'links' in source_kg else 'edges'
        for edge in source_kg.get(links_key, []):
            
            # Check input format
            src = edge.get('source')
            tgt = edge.get('target')
            
            if src in id_map and tgt in id_map:
                new_edge = {
                    'source': id_map[src],
                    'target': id_map[tgt]
                }
                
                # Avoid duplicate edges? 
                # Ideally yes, but list of dicts is hard to check efficiently without set.
                # For PoC, blind append is okay, but let's try to verify uniqueness if source/target match?
                # A simple check:
                # if new_edge not in current_edges: current_edges.append(new_edge)
                # But dict comparison works.
                
                # Avoid duplicate edges in global graph
                if new_edge not in current_edges:
                    current_edges.append(new_edge)
            
    # Merge Discovered
    # IMPORTANT: Simple append will duplicate nodes if we run loop multiple times. 
    # For PoC, let's just CLEAR the global graph and rebuild from laws list? 
    # Or just append. The user asked for "accumulating".
    # But if I verify, I restart loop. 
    # Let's assume loop clears accumulation dir or we handle it. 
    # Ideally, we check overlap. 
    # For now, simple append is fine, but maybe we should clear if we are starting a fresh task run?
    # The `loop` script appends. 
    # Let's just append.
    merge_into_graph("graph", new_kg)
    
    # Merge Ground Truth
    merge_into_graph("gt_graph", gt_kg)
    
    # --- Calculate OVERALL (Global) KG Sim ---
    global_kg["global_similarity"] = 0.0
    try:
        global_kg["global_similarity"] = kg.calculate_kg_similarity(global_kg["graph"], global_kg["gt_graph"])
    except Exception as e:
        print(f"Error calculating global similarity: {e}")
        
    # Save
    with open(global_kg_path, "w") as f:
        json.dump(global_kg, f, indent=2)
        
    print(f"Updated Global KG at {global_kg_path} (Global Sim: {global_kg['global_similarity']:.2f})")

def main():
    parser = argparse.ArgumentParser(description="Scientist Loop")
    parser.add_argument("--tasks", type=str, default="m0_gravity,m1_coulomb_force", help="Comma separated list of tasks or 'all'")
    parser.add_argument("--accumulation_dir", type=str, default="accumulation", help="Dir to store global state")
    parser.add_argument("--n_samples", type=int, default=300, help="Samples per task")
    parser.add_argument("--n_iterations", type=int, default=1000, help="SR iterations")
    parser.add_argument("--parsimony", type=float, default=0.001, help="Complexity penalty")
    parser.add_argument("--law_version", type=str, default=None, help="Specific law version (v0, v1, v2) or None for random")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    
    args = parser.parse_args()
    
    # 1. Determine Tasks
    if args.tasks == "all":
        tasks = adapter.get_available_tasks()
    else:
        tasks = args.tasks.split(",")
        
    # Filter only physics tasks (m*)
    tasks = [t for t in tasks if t.startswith("m")]
    
    print(f"Loop scheduled for {len(tasks)} tasks: {tasks}")
    
    # 2. Setup Accumulation Dir
    if os.path.exists(args.accumulation_dir):
        # Optional: Clear it to start fresh? 
        # User implies accumulating "start with simple... expand". 
        # But if we re-run, we might want to clear.
        # Let's NOT clear by default to allow resume, but `update_global_kg` handles duplicates poorly.
        # Let's clear for this PoC to avoid mess.
        shutil.rmtree(args.accumulation_dir)
        
    os.makedirs(args.accumulation_dir, exist_ok=True)
    
    # Initialize global KG
    global_kg_path = os.path.join(args.accumulation_dir, "global_kg.json")
    with open(global_kg_path, "w") as f:
        json.dump({
            "laws": [], 
            "graph": {"nodes": [], "edges": []},
            "gt_graph": {"nodes": [], "edges": []}
        }, f)
    
    # 3. Loop
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Starting discovery for: {task}")
        
        # Update Global KG with status
        if os.path.exists(global_kg_path):
            with open(global_kg_path, "r") as f:
                global_kg_data = json.load(f)
            global_kg_data["current_task"] = task
            global_kg_data["status"] = f"Discovering {task}..."
            with open(global_kg_path, "w") as f:
                json.dump(global_kg_data, f, indent=2)
        
        # Create a run config
        class Args:
            pass
            
        run_args = Args()
        run_args.task = task
        run_args.n_samples = args.n_samples
        run_args.seed = args.seed + i 
        run_args.noise = 0.0
        run_args.mode = "physics"
        run_args.output_dir = "runs"
        # Defaults for difficulty
        run_args.difficulty = "easy"
        run_args.law_version = args.law_version # User specified (or None for Random)
        run_args.n_iterations = args.n_iterations # Pass n_iterations
        run_args.parsimony = args.parsimony # Pass parsimony
        
        # Create output dir manually as run_physics_mode expects it passed
        run_id = f"physics_{task}_{run_args.seed}"
        run_dir = os.path.join("runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        config = vars(run_args)
        
        # Run
        try:
            run.run_physics_mode(run_args, run_dir, config)
            
            # Update Global KG
            update_global_kg(run_dir, args.accumulation_dir)
            
        except Exception as e:
            print(f"Failed task {task}: {e}")
            import traceback
            traceback.print_exc()
            
        # Simulate delay for visualization effect
        time.sleep(2)
        
    print("\n--- Scientist Loop Completed ---")

if __name__ == "__main__":
    main()
