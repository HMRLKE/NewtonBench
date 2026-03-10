import os
import re

formulas = {
    'm0_gravity': "    return (HIDDEN_CONSTANT * mass1 * mass2) / (distance ** 2)",
    'm1_coulomb_force': "    return (CONSTANT * q1 * q2) / (distance ** 2)",
    'm2_magnetic_force': "    return (CONSTANT * current1 * current2) / distance",
    'm3_fourier_law': "    try:\n        if d <= 0 or delta_T <= 0:\n            return 0.0\n        return (k * A * delta_T) / d\n    except (ValueError, ZeroDivisionError):\n        return float('nan')",
    'm4_snell_law': "    try:\n        return math.asin((n1 * math.sin(angle1)) / n2)\n    except (ValueError, ZeroDivisionError):\n        return float('nan')",
    'm5_radioactive_decay': "    try:\n        return N0 * math.exp(-lambda_constant * t)\n    except (ValueError, ZeroDivisionError, OverflowError):\n        return float('nan')",
    'm6_underdamped_harmonic': "    try:\n        return math.sqrt(k/m - (b/(2*m))**2)\n    except (ValueError, ZeroDivisionError):\n        return float('nan')",
    'm7_malus_law': "    try:\n        return I_0 * (math.cos(theta)) ** 2\n    except Exception:\n        return float('nan')",
    'm8_sound_speed': "    try:\n        return math.sqrt((gamma * HIDDEN_CONSTANT * T) / M)\n    except (ValueError, ZeroDivisionError):\n        return float('nan')",
    'm9_hooke_law': "    try:\n        if x < 0:\n            return float('nan')\n        return 0.5 * CONSTANT * (x ** 2)\n    except (ValueError, ZeroDivisionError):\n        return float('nan')",
    'm10_be_distribution': "    try:\n        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):\n            value = 1 / (np.exp(HIDDEN_CONSTANT * omega / T) - 1)\n        if value > 0 and np.isfinite(value):\n            return value\n        return float('nan')\n    except:\n        return float('nan')",
    'm11_heat_transfer': "    try:\n        return m * c * delta_T\n    except Exception:\n        return float('nan')"
}

sigs = {
    'm0_gravity': "def _ground_truth_law_v_unchanged(mass1: float, mass2: float, distance: float) -> float:",
    'm1_coulomb_force': "def _ground_truth_law_v_unchanged(q1: float, q2: float, distance: float) -> float:",
    'm2_magnetic_force': "def _ground_truth_law_v_unchanged(current1: float, current2: float, distance: float) -> float:",
    'm3_fourier_law': "def _ground_truth_law_v_unchanged(k: float, A: float, delta_T: float, d: float) -> float:",
    'm4_snell_law': "def _ground_truth_law_v_unchanged(n1: float, n2: float, angle1: float) -> float:",
    'm5_radioactive_decay': "def _ground_truth_law_v_unchanged(N0: float, lambda_constant: float, t: float) -> float:",
    'm6_underdamped_harmonic': "def _ground_truth_law_v_unchanged(k: float, m: float, b: float) -> float:",
    'm7_malus_law': "def _ground_truth_law_v_unchanged(I_0: float, theta: float) -> float:",
    'm8_sound_speed': "def _ground_truth_law_v_unchanged(gamma: float, T: float, M: float) -> float:",
    'm9_hooke_law': "def _ground_truth_law_v_unchanged(x: float) -> float:",
    'm10_be_distribution': "def _ground_truth_law_v_unchanged(omega: float, T: float) -> float:",
    'm11_heat_transfer': "def _ground_truth_law_v_unchanged(m: float, c: float, delta_T: float) -> float:"
}

for mod, formula in formulas.items():
    filepath = f"modules/{mod}/laws.py"
    if not os.path.exists(filepath):
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # insert v_unchanged function near the top of the laws section
    inject_str = f"{sigs[mod]}\n    \"\"\"Unchanged real-world law\"\"\"\n{formula}\n\n"
    if "def _ground_truth_law_v_unchanged" not in content:
        content = content.replace("# --- v0 laws ---", f"# --- v_unchanged law ---\n{inject_str}# --- v0 laws ---")
    
    # add to LAW_REGISTRY
    for difficulty in ['easy', 'medium', 'hard']:
        target = f"'{difficulty}': {{"
        if f"'{difficulty}': {{" in content:
            replace_str = f"'{difficulty}': {{\n        'v_unchanged': _ground_truth_law_v_unchanged,"
            if "'v_unchanged': _ground_truth_law_v_unchanged" not in content.split(target)[1].split('}')[0]:
                content = content.replace(target, replace_str)
                
    # update get_ground_truth_law signature
    old_sig = "def get_ground_truth_law(difficulty: str, law_version: Optional[str] = None) -> Tuple[Callable, str]:"
    new_sig = "def get_ground_truth_law(difficulty: str, law_version: Optional[str] = None, consistency: bool = False) -> Tuple[Callable, str]:"
    if old_sig in content:
        content = content.replace(old_sig, new_sig)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
print("Successfully injected v_unchanged and consistency args into all laws.py")
