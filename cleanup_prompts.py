import os

def cleanup_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    skip = False
    for i, line in enumerate(lines):
        # We want to remove:
        # if prompt_set == 'modified':
        #     prompts[0] = OBJECTIVE_PROMPT + "\nNote: This is a modified prompt set."
        
        if "if prompt_set == 'modified':" in line:
            # check if next line is the assignment
            if i + 1 < len(lines) and 'prompts[0] = OBJECTIVE_PROMPT' in lines[i+1]:
                skip = 2 # skip this and next
                continue
        
        if skip > 0:
            skip -= 1
            continue
            
        new_lines.append(line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

modules_dir = "modules"
for root, dirs, files in os.walk(modules_dir):
    if "prompts.py" in files:
        cleanup_file(os.path.join(root, "prompts.py"))

print("Cleaned up redundant prompt markers.")
