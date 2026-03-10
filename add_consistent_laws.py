import os
import re

formulas = {
    'm0_gravity': ("(HIDDEN_CONSTANT * mass1 * mass2)", "distance"),
    'm1_coulomb_force': ("(CONSTANT * q1 * q2)", "distance"),
    'm2_magnetic_force': ("(CONSTANT * current1 * current2)", "distance"),
}

sigs = {
    'm0_gravity': "def _ground_truth_law_{diff}_v{i}_consistent(mass1: float, mass2: float, distance: float) -> float:",
    'm1_coulomb_force': "def _ground_truth_law_{diff}_v{i}_consistent(q1: float, q2: float, distance: float) -> float:",
    'm2_magnetic_force': "def _ground_truth_law_{diff}_v{i}_consistent(current1: float, current2: float, distance: float) -> float:",
}

# we define structural transformations for the group: interaction_laws
# using shared_axis: distance_exponent
consistent_exponents = {
    'easy': {0: '1.5', 1: '2.5', 2: '3.0'},
    'medium': {0: '1.5', 1: '2.6', 2: '3.8'},
    'hard': {0: '2.718', 1: '2.6', 2: '3.8'}
}

for mod in formulas:
    filepath = f"modules/{mod}/laws.py"
    if not os.path.exists(filepath):
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # inject consistent laws for each difficulty
    new_laws_code = "\n# --- Consistent Laws (for --consistency flag) ---\n"
    
    for diff in ['easy', 'medium', 'hard']:
        for i in range(3):
            sig = sigs[mod].format(diff=diff, i=i)
            num, den = formulas[mod]
            exp = consistent_exponents[diff][i]
            
            # create the consistent function
            func = f"{sig}\n    \"\"\"Consistent {diff} v{i} law\"\"\"\n"
            func += "    try:\n"
            func += f"        if {den} <= 0:\n            return 0.0\n"
            func += f"        return {num} / ({den} ** {exp})\n"
            func += "    except (ValueError, ZeroDivisionError):\n"
            func += "        return float('nan')\n\n"
            new_laws_code += func
    
    # Add new block before LAW_REGISTRY
    if "# --- Consistent Laws (for --consistency flag) ---" not in content:
        content = content.replace("# --- Law Registry ---", f"{new_laws_code}# --- Law Registry ---")
        
    # Hook into get_ground_truth_law
    
    hook_str = """    if consistency and law_version in ['v0', 'v1', 'v2']:
        consistent_name = f"_ground_truth_law_{difficulty}_{law_version}_consistent"
        import sys
        mod_obj = sys.modules[__name__]
        if hasattr(mod_obj, consistent_name):
            consistent_func = getattr(mod_obj, consistent_name)
            return consistent_func, law_version
"""

    if "if consistency and law_version in ['v0', 'v1', 'v2']:" not in content:
        # insert hook before 'return LAW_REGISTRY[difficulty][selected_version], selected_version'
        old_return = "    return LAW_REGISTRY[difficulty][selected_version], selected_version"
        content = content.replace(old_return, f"{hook_str}\n{old_return}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print(f"Successfully injected _consistent laws for interaction_laws")
