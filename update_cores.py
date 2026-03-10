import os

def update_core_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update run_experiment_for_module signature
    run_sig_old = "    law_version: str = None,\n    **kwargs"
    run_sig_new = "    law_version: str = None,\n    consistency: bool = False,\n    **kwargs"
    
    if run_sig_old in content:
        content = content.replace(run_sig_old, run_sig_new)

    # Update get_ground_truth_law call in run_experiment_for_module
    call_old = "get_ground_truth_law(difficulty, law_version)"
    call_new = "get_ground_truth_law(difficulty, law_version, consistency)"
    
    if call_old in content:
        content = content.replace(call_old, call_new)

    # Update evaluate_law signature
    eval_sig_old = "    trial_info=None,\n) -> dict:"
    eval_sig_new = "    trial_info=None,\n    consistency: bool = False,\n) -> dict:"
    
    if eval_sig_old in content:
        content = content.replace(eval_sig_old, eval_sig_new)

    # Update get_ground_truth_law call in evaluate_law
    call_eval_old = "get_ground_truth_law(difficulty, law_version)"
    # Note: call_old and call_eval_old are usually same, replace handles both if present
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

modules_dir = "modules"
for root, dirs, files in os.walk(modules_dir):
    if "core.py" in files:
        update_core_file(os.path.join(root, "core.py"))

print("Updated core.py in all modules.")
