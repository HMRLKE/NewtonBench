import yaml
import os
from typing import Dict, Any, Callable

_CONSISTENCY_CONFIG = None

def get_consistency_config() -> Dict[str, Any]:
    global _CONSISTENCY_CONFIG
    if _CONSISTENCY_CONFIG is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'consistency_groups.yml')
        with open(config_path, 'r', encoding='utf-8') as f:
            _CONSISTENCY_CONFIG = yaml.safe_load(f)
    return _CONSISTENCY_CONFIG

def get_module_group(module_name: str) -> str:
    config = get_consistency_config()
    for group_name, group_data in config.items():
        if module_name in group_data.get('modules', []):
            return group_name
    return None

def override_law_if_consistent(
    module_name: str, 
    difficulty: str, 
    law_version: str, 
    original_law: Callable
) -> Callable:
    """
    If consistency is strictly requested, overrides the ground truth law
    by applying the canonical transformation of the conceptual group.
    """
    group_name = get_module_group(module_name)
    if not group_name:
        return original_law
        
    config = get_consistency_config()
    group_data = config.get(group_name, {})
    shared_axes = group_data.get('shared_axes', [])
    
    # If there are no shared axes, we cannot apply consistent transformations
    if not shared_axes:
        return original_law
        
    # Canonical transformations for interaction_laws
    if group_name == 'interaction_laws' and 'distance_exponent' in shared_axes:
        # We define a consistent set of distance exponents.
        # Original (v_unchanged): r^2 => exponent = 2
        # v0 easy: exponent = 1.5
        # v1 easy: exponent = 2.5
        # v2 easy: exponent = 3.0
        # ... (simplified for now as an example mapping)
        
        # This is where we will dynamically return a lambda that evaluates
        # the function with the consistent exponent instead of the hardcoded one.
        pass

    return original_law
