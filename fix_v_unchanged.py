import os
import re

sigs = {
    'm0_gravity': 'def _ground_truth_law_v_unchanged(mass1: float, mass2: float, distance: float) -> float:',
    'm1_coulomb_force': 'def _ground_truth_law_v_unchanged(q1: float, q2: float, distance: float) -> float:',
    'm2_magnetic_force': 'def _ground_truth_law_v_unchanged(current1: float, current2: float, distance: float) -> float:',
    'm3_fourier_law': 'def _ground_truth_law_v_unchanged(k: float, A: float, delta_T: float, d: float) -> float:',
    'm4_snell_law': 'def _ground_truth_law_v_unchanged(n1: float, n2: float, theta1: float) -> float:',
    'm5_radioactive_decay': 'def _ground_truth_law_v_unchanged(N0: float, lam: float, t: float) -> float:',
    'm6_underdamped_harmonic': 'def _ground_truth_law_v_unchanged(A: float, gamma: float, omega: float, t: float) -> float:',
    'm7_malus_law': 'def _ground_truth_law_v_unchanged(I0: float, theta: float) -> float:',
    'm8_sound_speed': 'def _ground_truth_law_v_unchanged(gamma: float, T: float, M: float) -> float:',
    'm9_hooke_law': 'def _ground_truth_law_v_unchanged(k: float, x: float) -> float:',
    'm10_be_distribution': 'def _ground_truth_law_v_unchanged(epsilon: float, mu: float, T: float) -> float:',
    'm11_heat_transfer': 'def _ground_truth_law_v_unchanged(m: float, c: float, delta_T: float) -> float:'
}

formulas = {
    'm0_gravity': '    C = 6.67430e-11\n    if distance <= 0:\n        return 0.0\n    return (C * mass1 * mass2) / (distance ** 2)',
    'm1_coulomb_force': '    K = 8.98755e9\n    if distance <= 0:\n        return 0.0\n    return (K * q1 * q2) / (distance ** 2)',
    'm2_magnetic_force': '    C = 2.0e-7\n    if distance <= 0:\n        return 0.0\n    return (C * current1 * current2) / distance',
    'm3_fourier_law': '    if d <= 0:\n        return 0.0\n    return (k * A * delta_T) / d',
    'm4_snell_law': '    return math.asin((n1 / n2) * math.sin(theta1))',
    'm5_radioactive_decay': '    return N0 * math.exp(-lam * t)',
    'm6_underdamped_harmonic': '    return A * math.exp(-gamma * t) * math.cos(omega * t)',
    'm7_malus_law': '    return I0 * (math.cos(theta) ** 2)',
    'm8_sound_speed': '    R = 8.314\n    return math.sqrt((gamma * R * T) / M)',
    'm9_hooke_law': '    return -k * x',
    'm10_be_distribution': '    k_B = 8.61733e-5\n    return 1.0 / (math.exp((epsilon - mu) / (k_B * T)) - 1)',
    'm11_heat_transfer': '    return m * c * delta_T'
}

def update_laws_file(mod, filepath):
    if mod not in sigs: return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "def _ground_truth_law_v_unchanged" not in content:
        # Find the first def _ground_truth_law and insert before it
        match = re.search(r'def _ground_truth_law_', content)
        if match:
            pos = match.start()
            # Try to find a header or blank line before it
            header_match = re.search(r'# --- .* ---', content[:pos][::-1])
            if header_match:
                header_pos = pos - header_match.end()
                pos = header_pos
            
            inject_str = f"# --- v_unchanged law ---\n{sigs[mod]}\n    \"\"\"Unchanged real-world law\"\"\"\n{formulas[mod]}\n\n"
            content = content[:pos] + inject_str + content[pos:]
            print(f"Injected function into {mod}")
        else:
            print(f"Could not find injection point in {mod}")
    
    # Ensure registration
    for difficulty in ['easy', 'medium', 'hard']:
        target = f"'{difficulty}': {{"
        if target in content:
            # Check if already there
            block_start = content.find(target)
            block_end = content.find('}', block_start)
            block = content[block_start:block_end]
            if "'v_unchanged': _ground_truth_law_v_unchanged" not in block:
                replace_str = f"'{difficulty}': {{\n        'v_unchanged': _ground_truth_law_v_unchanged,"
                content = content.replace(target, replace_str)
                print(f"Registered v_unchanged for {difficulty} in {mod}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

modules_dir = "modules"
for mod in sigs.keys():
    filepath = os.path.join(modules_dir, mod, "laws.py")
    if os.path.exists(filepath):
        update_laws_file(mod, filepath)
