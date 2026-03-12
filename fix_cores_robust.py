import os
import re

def fix_core_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Fix run_experiment_for_module signature
    # Match def run_experiment_for_module(...):
    # We want to insert 'consistency: bool = False,' before '**kwargs' or as a new param
    
    # Check if consistency is already in the signature
    run_match = re.search(r'def\s+run_experiment_for_module\s*\((.*?)\)\s*->', content, re.DOTALL)
    if not run_match:
        run_match = re.search(r'def\s+run_experiment_for_module\s*\((.*?)\)\s*:', content, re.DOTALL)
    
    if run_match:
        params = run_match.group(1)
        if 'consistency' not in params:
            # Add it before **kwargs or at the end
            if '**kwargs' in params:
                new_params = params.replace('**kwargs', 'consistency: bool = False,\n    **kwargs')
            else:
                new_params = params.rstrip() + ',\n    consistency: bool = False'
            content = content.replace(params, new_params)
            print(f"Added consistency to run_experiment_for_module in {filepath}")

    # 2. Fix evaluate_law signature
    eval_match = re.search(r'def\s+evaluate_law\s*\((.*?)\)\s*->', content, re.DOTALL)
    if not eval_match:
        eval_match = re.search(r'def\s+evaluate_law\s*\((.*?)\)\s*:', content, re.DOTALL)
        
    if eval_match:
        params = eval_match.group(1)
        if 'consistency' not in params:
            new_params = params.rstrip()
            if new_params.endswith(','):
                new_params += '\n    consistency: bool = False'
            else:
                new_params += ',\n    consistency: bool = False'
            content = content.replace(params, new_params)
            print(f"Added consistency to evaluate_law in {filepath}")

    # 3. Ensure get_ground_truth_law calls use consistency
    # Match get_ground_truth_law(diff, version) and replace with (diff, version, consistency)
    content = re.sub(r'get_ground_truth_law\s*\(\s*([^,]+)\s*,\s*([^,)]+)\s*\)', 
                     r'get_ground_truth_law(\1, \2, consistency)', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

modules_dir = "modules"
for root, dirs, files in os.walk(modules_dir):
    if "core.py" in files:
        fix_core_file(os.path.join(root, "core.py"))

print("Robust core.py fix complete.")
