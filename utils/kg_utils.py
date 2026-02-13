import sympy
import networkx as nx
from networkx.readwrite import json_graph
import os
import json
import time
import random

# Symbols to avoid name conflicts with SymPy functions
SYMPY_LOCALS = {
    'gamma': sympy.Symbol('gamma'),
    'beta': sympy.Symbol('beta'),
    'E': sympy.Symbol('E'),
    'N': sympy.Symbol('N'),
    'O': sympy.Symbol('O'),
    'I': sympy.Symbol('I'),
    'S': sympy.Symbol('S'),
    'lambda': sympy.Symbol('lambda_constant'),
    'exp': sympy.exp,
    'log': sympy.log,
    'sin': sympy.sin,
    'cos': sympy.cos,
    'tan': sympy.tan,
    'sqrt': sympy.sqrt,
    'asin': sympy.asin,
    'acos': sympy.acos,
    'atan': sympy.atan
}

def get_gt_equation_string(difficulty, law_version, task_name='m0_gravity'):
    """
    Returns the ground truth equation string for a given task and difficulty.
    Ported from mini_scientist/visualize.py
    """
    C = "6.674e-5"
    if 'coulomb' in task_name:
        C = "2.0"
        if difficulty == 'easy':
             if law_version == 'v0': return f"({C} * q1 * q2) / (distance**3)"
             if law_version == 'v1': return f"({C} * (q1 * q2)**3) / (distance**2)"
             if law_version == 'v2': return f"({C} * q1**3 * q2) / (distance**2)"
        return f"({C} * q1 * q2) / (distance**2)"
    if 'hooke' in task_name:
        K = "231.14"
        if difficulty == 'easy':
            if law_version == 'v1': return f"2 * {K} * x**0.5"
            if law_version == 'v2': return f"2 * {K} * x**3.4"
            return f"2 * {K} * x**2"
        return f"2 * {K} * x**2"
    if 'decay' in task_name:
        if difficulty == 'easy':
            if law_version == 'v1': return "N0 * exp(-lambda_constant**1.5 * t)"
            if law_version == 'v2': return "N0 * exp(-(lambda_constant * t)**1.5)"
            return "N0 * exp(-lambda_constant * t**1.5)"
        return "N0 * exp(-lambda_constant * t**1.5)"
    if 'fourier' in task_name:
        if difficulty == 'easy':
            if law_version == 'v1': return "(k * A**0.5 * delta_T) / d"
            if law_version == 'v2': return "(k * A * delta_T**2) / d"
            return "(k * A * delta_T) / d**2"
        return "(k * A * delta_T) / d**2"
    if 'snell' in task_name:
        if difficulty == 'easy':
            if law_version == 'v1': return "asin(n2 * sin(angle1) / n1)"
            if law_version == 'v2': return "atan(n1 * sin(angle1) / n2)"
            return "acos(n1 * sin(angle1) / n2)"
        return "acos(n1 * sin(angle1) / n2)"
    if 'underdamped' in task_name:
        if difficulty == 'easy':
            if law_version == 'v1': return "(k/m - (b/(2*m))**2)**2"
            if law_version == 'v2': return "k/m - (b/(2*m))**2"
            return "sqrt(k/m - b/(2*m))"
        return "sqrt(k/m - b/(2*m))"
    if 'sound' in task_name:
        C = "351.6"
        if difficulty == 'easy':
            if law_version == 'v1': return f"{C} * gamma * T / M"
            if law_version == 'v2': return f"sqrt({C} * T / M)"
            return f"sqrt({C} * gamma * T**2 / M)"
        return f"sqrt({C} * gamma * T**2 / M)"
    if 'be_distribution' in task_name:
        C = "1.0513e-14"
        if difficulty == 'easy':
            if law_version == 'v0': return f"1 / (exp({C} * omega / T) + 1)"
            if law_version == 'v1': return f"1 / (exp({C} * omega**0.5 / T) - 1)"
            if law_version == 'v2': return f"1 / (exp({C} * omega / T**3) - 1)"
        return f"1 / (exp({C} * omega / T) + 1)"
    if 'heat_transfer' in task_name:
        if difficulty == 'easy':
            if law_version == 'v0': return "m * c * (delta_T**2.5)"
            if law_version == 'v1': return "m**2.5 * c * delta_T"
            if law_version == 'v2': return "(m * delta_T)**2.5 * c"
        return "m * c * delta_T"
    if 'magnetic' in task_name:
        C = "2.0"
        return f"({C} * current1 * current2) / distance"
    if 'harmonic' in task_name and 'underdamped' not in task_name:
         return "sqrt(k/m)"
    elif 'gravity' in task_name or task_name == 'm0':
        if difficulty == 'easy':
            if law_version == 'v0': return f"({C} * mass1 * mass2) / (distance**1.5)"
            if law_version == 'v1': return f"({C} * mass1) / (distance**2)"
            if law_version == 'v2': return f"({C} * (mass1**2 * mass2**2)) / (distance**2)"
        elif difficulty == 'medium':
            if law_version == 'v0': return f"({C} * mass1 * mass2) / (distance**2.5)"
            if law_version == 'v1': return f"({C} * mass1**0.5 * mass2) / (distance**2)"
            if law_version == 'v2': return f"({C} * (mass1 * mass2)**0.5) / (distance**3)"
        elif difficulty == 'hard':
            if law_version == 'v0': return f"({C} * mass1 * mass2) / (distance**3.5)"
            if law_version == 'v1': return f"({C} * mass1**1.5 * mass2**1.5) / (distance**2)"
            if law_version == 'v2': return f"({C} * log(mass1 * mass2 + 1)) / (distance**2)"
        return f"({C} * mass1 * mass2) / (distance**2)"
    return "unknown"

def extract_expression(equation_str: str) -> str:
    """
    Attempts to extract just the equation expression from a potentially full Python function.
    """
    if "def discovered_law" in equation_str:
        # Try to find the 'return' statement
        lines = equation_str.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("return "):
                return line.replace("return ", "").strip()
    return equation_str

def equation_to_kg(equation_str: str) -> dict:
    """
    Converts a sympy equation string into a detailed knowledge graph (AST).
    Special handling for division to avoid nested negative powers.
    """
    equation_str = extract_expression(equation_str)
    try:
        expr = sympy.sympify(equation_str, locals=SYMPY_LOCALS, evaluate=False)
    except Exception as e:
        print(f"Error parsing equation '{equation_str}': {e}")
        return {"nodes": [], "edges": [], "graph": {"equation": equation_str, "error": str(e)}}
    
    G = nx.DiGraph()
    counter = 0
    node_lookup = {}
    
    def add_node(label, type="operator", sympy_node=None):
        nonlocal counter
        
        # Deduplication for non-operator nodes (vars/numbers)
        if type != "operator":
            key = (label, type)
            if key in node_lookup: return node_lookup[key]
        
        node_id = counter
        counter += 1
        G.add_node(node_id, label=label, type=type)
        
        if type != "operator":
            node_lookup[(label, type)] = node_id
        
        return node_id

    def add_edge(parent_id, child_id):
        if parent_id is not None:
             G.add_edge(parent_id, child_id)

    def process_node(node, parent_id=None):
        # 1. Handle Division (Mul with negative powers)
        if node.is_Mul:
            numerators = []
            denominators = []
            for arg in node.args:
                if arg.is_Pow and arg.exp.is_Number and arg.exp < 0:
                    denominators.append(arg)
                else:
                    numerators.append(arg)
            
            if denominators:
                # We have a division
                div_id = add_node("÷", "operator")
                add_edge(parent_id, div_id)
                
                # Handle Numerators
                if not numerators:
                    # e.g. 1/x -> numerators empty
                    one_id = add_node("1", "number")
                    add_edge(div_id, one_id)
                elif len(numerators) == 1:
                    process_node(numerators[0], div_id)
                else:
                    # Group multiple numerators under a *
                    mul_id = add_node("×", "operator")
                    add_edge(div_id, mul_id)
                    for n in numerators: process_node(n, mul_id)
                
                # Handle Denominators
                processed_denoms = []
                for d in denominators:
                     # d is Pow(base, neg_exp)
                     # Convert to positive exp
                     base, exp = d.base, d.exp
                     new_exp = -exp
                     if new_exp == 1:
                         processed_denoms.append(base)
                     else:
                         processed_denoms.append(sympy.Pow(base, new_exp, evaluate=False))
                
                if len(processed_denoms) == 1:
                     process_node(processed_denoms[0], div_id)
                else:
                     mul_id_denom = add_node("×", "operator")
                     add_edge(div_id, mul_id_denom)
                     for d in processed_denoms: process_node(d, mul_id_denom)
                return

        # 2. Handle 1/x (Pow with negative power) not in Mul
        if node.is_Pow and node.exp.is_Number and node.exp < 0:
             div_id = add_node("÷", "operator")
             add_edge(parent_id, div_id)
             
             one_id = add_node("1", "number")
             add_edge(div_id, one_id)
             
             base, exp = node.base, node.exp
             new_exp = -exp
             if new_exp == 1:
                 process_node(base, div_id)
             else:
                 process_node(sympy.Pow(base, new_exp, evaluate=False), div_id)
             return

        # 3. Standard Handling
        val = str(node)
        is_op = False
        is_var = False
        is_num = False
        
        if node.is_Add: val = "+"; is_op = True
        elif node.is_Mul: val = "×"; is_op = True
        elif node.is_Pow: val = "^"; is_op = True
        elif node.is_Function: val = str(node.func); is_op = True
        elif node.is_Symbol: is_var = True
        elif node.is_Number:
            is_num = True
            try:
                f_val = float(node)
                val = f"{f_val:.3g}"
                if "e" not in val and "." in val: val = val.rstrip("0").rstrip(".")
            except: val = str(node)
        
        node_type = "operator" if is_op else "variable" if is_var else "number"
        
        current_id = add_node(val, node_type)
        add_edge(parent_id, current_id)
        
        if hasattr(node, "args"):
             for arg in node.args: process_node(arg, current_id)

    process_node(expr)
    G.graph["equation"] = equation_str
    return json_graph.node_link_data(G)

def calculate_kg_similarity(g1: dict, g2: dict) -> float:
    """
    Calculates a similarity score [0, 1] between two knowledge graphs.
    Ported from mini_scientist/kg.py
    """
    nodes1 = g1.get('nodes', [])
    nodes2 = g2.get('nodes', [])
    if not nodes1 or not nodes2: return 0.0
    from collections import Counter
    def get_node_counts(nodes):
        ops, vars, nums = Counter(), Counter(), Counter()
        for n in nodes:
            t, l = n.get('type'), str(n.get('label'))
            if t == 'operator': ops[l] += 1
            elif t == 'variable': vars[l] += 1
            elif t == 'number': nums[l] += 1
        return ops, vars, nums
    ops1, vars1, nums1 = get_node_counts(nodes1)
    ops2, vars2, nums2 = get_node_counts(nodes2)
    def multiset_jaccard(c1, c2):
        if not c1 and not c2: return 1.0
        intersection = sum((c1 & c2).values())
        union = sum((c1 | c2).values())
        return intersection / union if union > 0 else 0.0
    def fuzzy_jaccard_nums(n1_list, n2_list):
        if not n1_list and not n2_list: return 1.0
        matches, used_j = [], set()
        f1 = [float(x) if x.replace('.','',1).replace('e','',1).replace('-','',1).isdigit() else 0.0 for x in n1_list]
        f2 = [float(x) if x.replace('.','',1).replace('e','',1).replace('-','',1).isdigit() else 0.0 for x in n2_list]
        for v1 in f1:
            best_sim, best_j = 0, -1
            for j, v2 in enumerate(f2):
                if j in used_j: continue
                denom = max(abs(v1), abs(v2), 1e-9)
                sim = max(0, 1 - abs(v1 - v2) / denom)
                if sim > best_sim: best_sim = sim; best_j = j
            if best_j != -1 and best_sim > 0.8: matches.append(best_sim); used_j.add(best_j)
        inter = sum(matches)
        union = len(f1) + len(f2) - len(matches)
        return inter / union if union > 0 else 0.0
    def counter_to_list(c):
        res = []
        for k, v in c.items(): res.extend([k]*v)
        return res
    s_ops = multiset_jaccard(ops1, ops2)
    s_vars = multiset_jaccard(vars1, vars2)
    s_nums = fuzzy_jaccard_nums(counter_to_list(nums1), counter_to_list(nums2))
    def get_edge_counts(g):
        nodes = {n['id']: n for n in g.get('nodes', [])}
        edges = Counter()
        links_key = 'links' if 'links' in g else 'edges'
        for e in g.get(links_key, []):
            s_id, t_id = e.get('source'), e.get('target')
            if s_id in nodes and t_id in nodes:
                key = (nodes[s_id]['label'], nodes[t_id]['label'])
                edges[key] += 1
        return edges
    s_edges = multiset_jaccard(get_edge_counts(g1), get_edge_counts(g2))
    return (s_ops * 0.15 + s_vars * 0.35 + s_nums * 0.05 + s_edges * 0.45)

def update_global_dashboard(trial_id, module_name, equation, difficulty, law_version, metrics, chat_history=None, accumulation_dir="accumulation"):
    """
    Updates global_kg.json with results from the latest trial.
    Only updates if the new result is "better" (lower RMSLE or higher Accuracy) than existing.
    """
    if chat_history is None: chat_history = []
    
    # helper for cleaning equation
    def clean_equation_for_display(eq_str):
        # 1. Remove def ... return
        if "def discovered_law" in eq_str:
            lines = eq_str.split("\n")
            for line in reversed(lines):
                if line.strip().startswith("return "):
                    eq_str = line.strip().replace("return ", "")
                    break
        # 2. Basic cleanup for MathJax (python to latex-ish)
        return eq_str

    if not os.path.exists(accumulation_dir):
        os.makedirs(accumulation_dir)
        
    global_kg_path = os.path.join(accumulation_dir, "global_kg.json")
    print(f"[DEBUG] update_global_dashboard called. Writing to: {os.path.abspath(global_kg_path)}")
    
    # Retry mechanism for file locking
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Initialize basic structure
            global_kg = {
                "status": "Online (Polling)",
                "laws": [],
                "graph": {"nodes": [], "edges": []},
                "gt_graph": {"nodes": [], "edges": []},
                "global_similarity": 0.0
            }
            
            if os.path.exists(global_kg_path):
                try:
                    with open(global_kg_path, "r") as f:
                        data = json.load(f)
                        if data: global_kg = data
                except json.JSONDecodeError:
                    pass # Start fresh if corrupted

            # Check if we should update this task
            existing_law_idx = -1
            for i, law in enumerate(global_kg["laws"]):
                if law["task"] == module_name and law["difficulty"] == difficulty and law["version"] == law_version:
                    existing_law_idx = i
                    break
            
            new_rmsle = metrics.get("rmsle", float("inf"))
            new_acc = metrics.get("exact_accuracy", 0.0)
            
            should_update = False
            if existing_law_idx == -1:
                should_update = True
            else:
                existing_law = global_kg["laws"][existing_law_idx]
                old_rmsle = existing_law.get("rmsle", float("inf"))
                old_acc = existing_law.get("exact_accuracy", 0.0) # Backwards compat
                
                # Update if better accuracy OR (same accuracy AND better RMSLE)
                if new_acc > old_acc:
                    should_update = True
                elif new_acc == old_acc and new_rmsle < old_rmsle:
                    should_update = True
            
            if not should_update:
                print(f"[Dashboard] Skipping update for {module_name} (New RMSLE: {new_rmsle:.4f} not better than existing)")
                return

            # Prepare new law entry
            gt_eqn_str = get_gt_equation_string(difficulty=difficulty, law_version=law_version, task_name=module_name)
            clean_eq = clean_equation_for_display(equation)

            new_law = {
                "task": module_name,
                "difficulty": difficulty,
                "version": law_version,
                "equation": clean_eq, # Cleaned for display
                "raw_equation": equation, # Keep raw just in case
                "gt_equation": gt_eqn_str, 
                "loss": metrics.get("rmsle", 0), # Use RMSLE as loss since RMSE isn't returned
                "rmsle": metrics.get("rmsle", 0),
                "exact_accuracy": metrics.get("exact_accuracy", 0.0),
                "similarity": metrics.get("kg_similarity", 0),
                "symbolic_match": metrics.get("symbolic_equivalent", False),
                "chat_history": chat_history # Add history
            }

            try:
                trial_kg = equation_to_kg(clean_eq)
            except Exception as e:
                print(f"Error parsing trial equation: {e}")
                trial_kg = {"nodes": [], "edges": []}

            try:
                gt_kg = equation_to_kg(gt_eqn_str)
            except Exception as e:
                # Expected if gt is unknown or parsing fails
                gt_kg = {"nodes": [], "edges": []}
            
            
            # --- UPDATE GLOBAL STATE ---
            if existing_law_idx != -1:
                global_kg["laws"][existing_law_idx] = new_law
            else:
                global_kg["laws"].append(new_law)
            
            # For now, dashboard graph view shows ONLY the current updated law
            global_kg["graph"] = trial_kg
            global_kg["gt_graph"] = gt_kg
            
            # Recalculate global similarity
            sim_score = calculate_kg_similarity(trial_kg, gt_kg)
            new_law["similarity"] = sim_score
            global_kg["global_similarity"] = sim_score

            with open(global_kg_path, "w") as f:
                json.dump(global_kg, f, indent=2)
                
            break # Success
            
        except IOError:
            time.sleep(random.uniform(0.1, 0.3))
        except Exception as e:
             print(f"Error updating dashboard: {e}")
             break
def merge_into_graph(global_kg, graph_key, source_kg, task_name):
    if graph_key not in global_kg: global_kg[graph_key] = {"nodes": [], "edges": []}
    current_nodes, current_edges = global_kg[graph_key]["nodes"], global_kg[graph_key]["edges"]
    node_lookup = {}
    for n in current_nodes:
        if n['type'] in ['variable', 'number']:
            key = (n['label'], n['type']); node_lookup[key] = n['id']
    current_max_id = max([n['id'] for n in current_nodes] + [-1]) + 1
    id_map = {}
    for node in source_kg.get('nodes', []):
        old_id, n_type, n_label = node['id'], node.get('type', 'default'), str(node.get('label', node.get('id')))
        key = (n_label, n_type)
        if n_type in ['variable', 'number'] and key in node_lookup: id_map[old_id] = node_lookup[key]
        else:
            new_id = current_max_id; current_max_id += 1; id_map[old_id] = new_id
            n_copy = node.copy(); n_copy['id'] = new_id; n_copy['task'] = task_name
            current_nodes.append(n_copy)
            if n_type in ['variable', 'number']: node_lookup[key] = new_id
    links_key = 'links' if 'links' in source_kg else 'edges'
    for edge in source_kg.get(links_key, []):
        src, tgt = edge.get('source'), edge.get('target')
        if src in id_map and tgt in id_map:
            new_edge = {'source': id_map[src], 'target': id_map[tgt]}
            if new_edge not in current_edges: current_edges.append(new_edge)

