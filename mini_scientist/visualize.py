import os
import json
import argparse
import sys
import webbrowser
import sympy

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_scientist import kg
# We need to access laws registry to regenerate the GT KG
from modules.m0_gravity.laws import LAW_REGISTRY, HIDDEN_CONSTANT

def get_gt_equation_string(difficulty, law_version, task_name='m0_gravity'):
    # Default C for m0
    C = "6.674e-5"
    
    if 'coulomb' in task_name:
        C = "2.0"
        if difficulty == 'easy':
             if law_version == 'v0': return f"({C} * q1 * q2) / (distance**3)"
             if law_version == 'v1': return f"({C} * (q1 * q2)**3) / (distance**2)"
             if law_version == 'v2': return f"({C} * q1**3 * q2) / (distance**2)"
        return f"({C} * q1 * q2) / (distance**2)"

    if 'hooke' in task_name:
        # Easy Hooke's law: U = 2kx^2 (v0)
        K = "231.14"
        if difficulty == 'easy':
            if law_version == 'v1': return f"2 * {K} * x**0.5"
            if law_version == 'v2': return f"2 * {K} * x**3.4"
            return f"2 * {K} * x**2"
        return f"2 * {K} * x**2"

    if 'decay' in task_name:
        # N(t) = N₀ * e^(-λ * t^1.5)
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
        # Easy Snell's law variant
        if difficulty == 'easy':
            if law_version == 'v1': return "asin(n2 * sin(angle1) / n1)"
            if law_version == 'v2': return "atan(n1 * sin(angle1) / n2)"
            return "acos(n1 * sin(angle1) / n2)"
        return "acos(n1 * sin(angle1) / n2)"

    if 'underdamped' in task_name:
        # ω = k/m - (b/(2*m))
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
        # F = (C * i1 * i2) / distance
        return f"({C} * current1 * current2) / distance"

    if 'harmonic' in task_name and 'underdamped' not in task_name:
         # Fallback for simple harmonic if any
         return "sqrt(k/m)"

    # m0_gravity (Original Logic)
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
        
        # Default Gravity
        return f"({C} * mass1 * mass2) / (distance**2)"

    return "unknown"

def generate_html(gt_kg, discovered_kg, output_path):
    """
    Generates a single HTML file with two network visualizations using vis.js.
    """
    
    gt_nodes = json.dumps(gt_kg['nodes'])
    gt_edges = json.dumps(gt_kg['link_data']['links'] if 'link_data' in gt_kg else gt_kg['edges'])
    
    # LaTeX Conversion
    gt_eqn_str = gt_kg.get('graph', {}).get('equation', '')
    disc_eqn_str = discovered_kg.get('graph', {}).get('equation', '')
    
    try:
        gt_latex = sympy.latex(sympy.sympify(gt_eqn_str, locals=kg.SYMPY_LOCALS, evaluate=False))
    except:
        gt_latex = str(gt_eqn_str)
        
    try:
        disc_latex = sympy.latex(sympy.sympify(disc_eqn_str, locals=kg.SYMPY_LOCALS, evaluate=False))
    except:
        disc_latex = str(disc_eqn_str)

    def normalize_graph_data(g):
        nodes = []
        for n in g['nodes']:
            # Vis.js expects 'label'
            node = {'id': n['id'], 'label': str(n.get('label', n.get('id'))), 'group': n.get('type', 'default')}
            nodes.append(node)
            
        edges = []
        links_key = 'links' if 'links' in g else 'edges'
        for e in g[links_key]:
            edges.append({'from': e['source'], 'to': e['target'], 'arrows': 'to'})
            
        return nodes, edges

    gt_nodes_data, gt_edges_data = normalize_graph_data(gt_kg)
    disc_nodes_data, disc_edges_data = normalize_graph_data(discovered_kg)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style type="text/css">
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; align-items: center; background-color: #fcece8; }}
        h1 {{ color: #a4303f; }}
        .equation-box {{ 
            background: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
        }}
        .container {{ display: flex; width: 95%; justify-content: space-around; }}
        .graph-container {{ width: 45%; background: white; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        #gt-network, #disc-network {{ width: 100%; height: 500px; border: 1px solid #ddd; background: #fafafa; }}
        h2 {{ text-align: center; color: #555; }}
        .legend {{ text-align: center; margin-bottom: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Equation Knowledge Graph Comparison</h1>
    <div class="legend">
        <span style="color: #2B7CE9">Variables</span> | 
        <span style="color: #FF5555">Operators</span> | 
        <span style="color: #55FF55">Constants</span>
    </div>
    
    <div class="container">
        <div class="graph-container">
            <h2>Ground Truth (Hidden Law)</h2>
            <div class="equation-box">
                $$ {gt_latex} $$
                <small style="color:gray">{gt_eqn_str}</small>
            </div>
            <div id="gt-network"></div>
        </div>
        
        <div class="graph-container">
            <h2>Discovered Law (Explored)</h2>
             <div class="equation-box">
                $$ {disc_latex} $$
                <small style="color:gray">{disc_eqn_str}</small>
            </div>
            <div id="disc-network"></div>
        </div>
    </div>

    <script type="text/javascript">
        // Common Options
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 20,
                font: {{ size: 16 }}
            }},
            edges: {{
                width: 2,
                font: {{ align: 'middle' }},
                smooth: {{ type: 'cubicBezier' }}
            }},
            groups: {{
                variable: {{ color: {{ background: '#97C2FC', border: '#2B7CE9' }} }},
                operator: {{ color: {{ background: '#FF9999', border: '#FF5555' }} }},
                number: {{ color: {{ background: '#99FF99', border: '#55FF55' }} }}
            }},
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 100,
                    nodeSpacing: 100
                }}
            }},
            physics: {{
                hierarchicalRepulsion: {{
                    nodeDistance: 120
                }}
            }}
        }};

        // Ground Truth Graph
        var gtNodes = new vis.DataSet({json.dumps(gt_nodes_data)});
        var gtEdges = new vis.DataSet({json.dumps(gt_edges_data)});
        var gtContainer = document.getElementById('gt-network');
        var gtData = {{ nodes: gtNodes, edges: gtEdges }};
        var gtNetwork = new vis.Network(gtContainer, gtData, options);

        // Discovered Graph
        var discNodes = new vis.DataSet({json.dumps(disc_nodes_data)});
        var discEdges = new vis.DataSet({json.dumps(disc_edges_data)});
        var discContainer = document.getElementById('disc-network');
        var discData = {{ nodes: discNodes, edges: discEdges }};
        var discNetwork = new vis.Network(discContainer, discData, options);
    </script>
</body>
</html>
    """
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    return os.path.abspath(output_path)

def main():
    parser = argparse.ArgumentParser(description="Visualize Ground Truth vs Discovered KGs")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory")
    args = parser.parse_args()
    
    run_dir = args.run_dir
    
    # 1. Load Config to find GT
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {run_dir}")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    difficulty = config.get('difficulty', 'easy')
    law_version = config.get('law_version')
    
    if law_version is None:
        print("Warning: 'law_version' not found in config. Assuming 'v0' or random choice made by system previously.")
        # Try to infer or just warn
        law_version = 'v0' 
        
    print(f"Run Configuration: Difficulty={difficulty}, Version={law_version}")
    
    # 2. Reconstruct GT KG
    gt_eqn_str = get_gt_equation_string(difficulty, law_version)
    print(f"Ground Truth Equation: {gt_eqn_str}")
    gt_kg = kg.equation_to_kg(gt_eqn_str)
    
    # 3. Load Discovered KG
    kg_path = os.path.join(run_dir, "kg.json")
    if not os.path.exists(kg_path):
        print(f"Error: kg.json not found in {run_dir}")
        return
        
    with open(kg_path, "r") as f:
        src_kg = json.load(f)
        
    # Check if this loading is correct 
    # (previous view showed 'graph' key wrapper? No, json_graph.node_link_data returns dict with 'nodes', 'links'/'edges', 'graph')
    discovered_kg = src_kg
    
    # 4. Generate Visualization
    output_html = os.path.join(run_dir, "visualization.html")
    abs_path = generate_html(gt_kg, discovered_kg, output_html)
    
    print(f"Visualization generated at: {abs_path}")
    
    # Open in browser
    # webbrowser.open(f"file://{abs_path}")

if __name__ == "__main__":
    main()
