import sympy
import networkx as nx
from networkx.readwrite import json_graph

# Symbols to avoid name conflicts with SymPy functions (e.g., gamma, beta, E)
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

def equation_to_kg(equation_str: str) -> dict:
    """
    Converts a sympy equation string into a detailed knowledge graph (AST).
    
    Args:
        equation_str: The equation as a string.
        
    Returns:
        A dictionary representation of the NetworkX graph.
    """
    expr = sympy.sympify(equation_str, locals=SYMPY_LOCALS, evaluate=False)
    
    G = nx.DiGraph()
    
    counter = 0
    node_lookup = {} # To reuse variables and numbers
    
    def get_or_create_node(node):
        nonlocal counter
        
        # Determine properties
        is_op = False
        is_var = False
        is_num = False
        val = str(node)
        
        if node.is_Add:
            val = "+"
            is_op = True
        elif node.is_Mul:
            val = "×"
            is_op = True
        elif node.is_Pow:
            val = "^"
            is_op = True
        elif node.is_Function:
            val = str(node.func)
            is_op = True
        elif node.is_Symbol:
            is_var = True
        elif node.is_Number:
            is_num = True
            try:
                # Round to 3 decimals max, removing trailing zeros
                f_val = float(node)
                val = f"{f_val:.3g}"
                if "e" not in val and "." in val:
                     val = val.rstrip("0").rstrip(".")
            except:
                val = str(node)
        
        node_type = "operator" if is_op else "variable" if is_var else "number"
        
        # Concepts (variables/numbers) should be reused. Operators should be UNIQUE per occurrence in AST.
        if not is_op:
            key = (val, node_type)
            if key in node_lookup:
                return node_lookup[key]
            
        node_id = counter
        counter += 1
        
        G.add_node(node_id, label=val, type=node_type)
        
        if not is_op:
            node_lookup[(val, node_type)] = node_id
            
        return node_id
        
    def traverse(node, parent_id=None):
        current_id = get_or_create_node(node)
        
        if parent_id is not None:
            # Avoid duplicate edges if we reuse nodes
            if not G.has_edge(parent_id, current_id):
                G.add_edge(parent_id, current_id)
            
        if hasattr(node, "args"):
            for arg in node.args:
                traverse(arg, current_id)
                
    traverse(expr)
    
    # Add metadata
    G.graph["equation"] = equation_str
    
    return json_graph.node_link_data(G)

def calculate_kg_similarity(g1: dict, g2: dict) -> float:
    """
    Calculates a similarity score [0, 1] between two knowledge graphs.
    Uses Jaccard-like overlap of nodes and edges.
    Constants (numbers) are matched fuzzily.
    """
    nodes1 = g1.get('nodes', [])
    nodes2 = g2.get('nodes', [])
    
    if not nodes1 or not nodes2:
        return 0.0

    # 1. Node Similarity
    from collections import Counter
    
    def get_node_counts(nodes):
        ops, vars, nums = Counter(), Counter(), Counter()
        for n in nodes:
            t = n.get('type')
            l = str(n.get('label'))
            if t == 'operator': ops[l] += 1
            elif t == 'variable': vars[l] += 1
            elif t == 'number': nums[l] += 1
        return ops, vars, nums

    ops1, vars1, nums1 = get_node_counts(nodes1)
    ops2, vars2, nums2 = get_node_counts(nodes2)

    def multiset_jaccard(c1, c2):
        if not c1 and not c2: return 1.0
        if not c1 or not c2: return 0.0
        intersection = sum((c1 & c2).values())
        union = sum((c1 | c2).values())
        return intersection / union if union > 0 else 0.0

    s_ops = multiset_jaccard(ops1, ops2)
    s_vars = multiset_jaccard(vars1, vars2)
    
    # Fuzzy Jaccard for Numbers
    def fuzzy_jaccard_nums(n1_list, n2_list):
        if not n1_list and not n2_list: return 1.0
        if not n1_list or not n2_list: return 0.0
        
        matches = []
        used_j = set()
        
        def to_f(s):
            try: return float(s)
            except: return 0.0
        
        f1 = [to_f(x) for x in n1_list]
        f2 = [to_f(x) for x in n2_list]
        
        for v1 in f1:
            best_sim = 0
            best_j = -1
            for j, v2 in enumerate(f2):
                if j in used_j: continue
                denom = max(abs(v1), abs(v2), 1e-9)
                sim = max(0, 1 - abs(v1 - v2) / denom)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            
            if best_j != -1 and best_sim > 0.8: # Stricter threshold for numbers
                matches.append(best_sim)
                used_j.add(best_j)
        
        inter = sum(matches)
        union = len(f1) + len(f2) - len(matches)
        return inter / union if union > 0 else 0.0

    # For fuzzy jaccard we need lists
    def counter_to_list(c):
        res = []
        for k, v in c.items(): res.extend([k]*v)
        return res

    s_nums = fuzzy_jaccard_nums(counter_to_list(nums1), counter_to_list(nums2))

    # 2. Edge Similarity
    def get_edge_counts(g):
        nodes = {n['id']: n for n in g.get('nodes', [])}
        edges = Counter()
        links_key = 'links' if 'links' in g else 'edges'
        for e in g.get(links_key, []):
            s_id = e.get('source')
            t_id = e.get('target')
            if s_id in nodes and t_id in nodes:
                key = (nodes[s_id]['label'], nodes[t_id]['label'])
                edges[key] += 1
        return edges

    edges1 = get_edge_counts(g1)
    edges2 = get_edge_counts(g2)
    s_edges = multiset_jaccard(edges1, edges2)

    # Weighted Average: Increase weight of variables and structure
    return (s_ops * 0.15 + s_vars * 0.35 + s_nums * 0.05 + s_edges * 0.45)

