import os
import glob

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Update function signature
    old_sig = "def get_task_prompt(system: str, is_code_assisted: bool = False, noise_level: float = 0.0) -> str:"
    new_sig = "def get_task_prompt(system: str, is_code_assisted: bool = False, noise_level: float = 0.0, prompt_set: str = 'original') -> str:"
    
    if old_sig in content:
        content = content.replace(old_sig, new_sig)
        
    old_body = "	prompts = [OBJECTIVE_PROMPT]\n\n	if noise_level > 0.0:"
    new_body = "	prompts = [OBJECTIVE_PROMPT]\n\n	if prompt_set == 'modified':\n		prompts[0] = OBJECTIVE_PROMPT + \"\\nNote: This is a modified prompt set.\"\n\n	if noise_level > 0.0:"
    
    old_body_spaces = "    prompts = [OBJECTIVE_PROMPT]\n\n    if noise_level > 0.0:"
    new_body_spaces = "    prompts = [OBJECTIVE_PROMPT]\n\n    if prompt_set == 'modified':\n        prompts[0] = OBJECTIVE_PROMPT + \"\\nNote: This is a modified prompt set.\"\n\n    if noise_level > 0.0:"
    
    if old_body in content:
        content = content.replace(old_body, new_body)
    elif old_body_spaces in content:
        content = content.replace(old_body_spaces, new_body_spaces)
        
    with open(filepath, 'w') as f:
        f.write(content)

modules_dir = r"c:\Users\Hamerlik\Documents\GitHub\hw2h\NewtonBench\modules"
for root, dirs, files in os.walk(modules_dir):
    for file in files:
        if file == "prompts.py":
            update_file(os.path.join(root, file))

print("Done updating prompts.py in all modules.")
