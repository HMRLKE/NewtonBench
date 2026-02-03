import importlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import os
import random

# Explicit registry to handle modules using **kwargs
MODULE_REGISTRY = {
    'm0_gravity': ['mass1', 'mass2', 'distance'],
    'm1_coulomb_force': ['q1', 'q2', 'distance'],
    'm2_magnetic_force': ['current1', 'current2', 'distance'],
    'm3_fourier_law': ['k', 'A', 'delta_T', 'd'],
    'm4_snell_law': ['refractive_index_1', 'refractive_index_2', 'incidence_angle'],
    'm5_radioactive_decay': ['N0', 'lambda_constant', 't'],
    'm6_underdamped_harmonic': ['k_constant', 'mass', 'b_constant'],
    'm7_malus_law': ['I_0', 'theta'],
    'm8_sound_speed': ['adiabatic_index', 'temperature', 'molar_mass'],
    'm9_hooke_law': ['x', 'm'],
    'm10_be_distribution': ['omega', 'temperature'],
    'm11_heat_transfer': ['m', 'c', 'delta_T']
}

PARAMETER_RANGES = {
    'mass1': (1.0, 10.0), 'mass2': (1.0, 10.0), 'mass': (1.0, 10.0), 'm': (1.0, 10.0),
    'distance': (1.0, 5.0), 'd': (1.0, 5.0), 'r': (1.0, 5.0), 'x': (0.1, 2.0),
    'q1': (1.0, 10.0), 'q2': (1.0, 10.0),
    'current1': (1.0, 10.0), 'current2': (1.0, 10.0),
    'k': (1.0, 5.0), 'k_constant': (1.0, 10.0), 'b_constant': (0.1, 2.0),
    'A': (1.0, 10.0), 'delta_T': (1.0, 10.0),
    'N0': (100.0, 1000.0), 'lambda_constant': (0.01, 0.5), 't': (1.0, 10.0),
    'refractive_index_1': (1.0, 1.5), 'refractive_index_2': (1.5, 2.0), 'incidence_angle': (0.0, 60.0),
    'I_0': (100.0, 1000.0), 'theta': (0.0, 1.5), # radians
    'adiabatic_index': (1.3, 1.7), 'temperature': (10.0, 1000.0), 'molar_mass': (0.02, 0.05),
    'omega': (1e13, 1e17), 'c': (100.0, 1000.0)
}

def get_available_tasks() -> List[str]:
    """Lists available NewtonBench tasks (modules)."""
    return sorted(list(MODULE_REGISTRY.keys()))

def get_sampling_params(task_name: str) -> List[str]:
    """Returns the list of input parameters for a given task."""
    return MODULE_REGISTRY.get(task_name, [])

def generate_data(task_name: str, n_samples: int = 100, noise: float = 0.0, difficulty: str = 'easy', law_version: str = 'v0') -> pd.DataFrame:
    """Generates a dataset for a given NewtonBench task."""
    try:
        module_path = f"modules.{task_name}.core"
        mod = importlib.import_module(module_path)
    except ImportError:
        print(f"Error: Could not import module {module_path}")
        return pd.DataFrame()

    params_to_sample = get_sampling_params(task_name)
    if not params_to_sample:
        print(f"Warning: No parameters found for task {task_name}")
        return pd.DataFrame()

    data = []
    for _ in range(n_samples):
        sample = {}
        for p in params_to_sample:
            low, high = PARAMETER_RANGES.get(p, (1.0, 10.0))
            # Use log-uniform if the range is large
            if high / low > 100:
                sample[p] = np.exp(random.uniform(np.log(low), np.log(high)))
            else:
                sample[p] = random.uniform(low, high)
        
        try:
            # Use explicit kwargs for the experiment runner
            y = mod.run_experiment_for_module(
                noise_level=noise,
                difficulty=difficulty,
                law_version=law_version,
                **sample
            )
            
            # Handle dictionary results (some modules return time series)
            if isinstance(y, dict):
                 # Find first numeric list that isn't 'time' or 'x'
                 target_col = None
                 for k, v in y.items():
                     if k not in ['time', 'x'] and isinstance(v, list) and len(v) > 0:
                         target_col = k
                         break
                 
                 if target_col:
                     # Take the last measurement as the representative value for SR
                     y_val = float(y[target_col][-1])
                 else:
                     y_val = 0.0
            else:
                y_val = float(y)
                
            sample['target'] = y_val
            data.append(sample)
        except Exception:
            # Skip invalid samples
            continue

    return pd.DataFrame(data)
