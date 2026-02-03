import pandas as pd
from pysr import PySRRegressor
import sympy
import os

def run_sr(X: pd.DataFrame, y: pd.Series, n_iterations: int = 1000, parsimony: float = 0.001, temp_dir: str = None) -> dict:
    """
    Runs symbolic regression on the data.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_iterations: Number of iterations for PySR.
        parsimony: Complexity penalty (higher = simpler equations).
        temp_dir: Directory to store PySR temp files.
        
    Returns:
        Dictionary containing the best equation string, sympy expression, and metrics.
    """
    # Configure PySR
    model = PySRRegressor(
        niterations=n_iterations,  # Increased for better convergence
        binary_operators=["+", "*", "-", "/", "^"], 
        unary_operators=["exp", "log", "sin", "cos"],
        extra_sympy_mappings={"square": lambda x: x**2},
        constraints={'^': (-5, 5)}, # Prevent extreme powers
        nested_constraints={"^": {"^": 1}}, # No power of a power
        parsimony=parsimony, # Stronger bias towards simpler laws
        maxsize=25, # Prevent overly long / nonsense equations
        model_selection="best",
        tempdir=temp_dir,
        verbosity=1,
        progress=False
    )
    
    model.fit(X, y)
    
    # Get best equation
    # PySR stores equations in .equations_ attribute (DataFrame)
    best_idx = model.equations_.iloc[-1].name # Last one is usually best by some metric in 'best' selection
    # But let's rely on model.sympy() which returns the best one
    
    best_eqn = model.sympy()
    eqn_str = str(best_eqn)
    
    # Retrieve metrics for the best equation
    # model.equations_ is a dataframe with idx as index. We need to find the one matching best_eqn
    # A simple way is to take the row with highest score or just the one selected.
    # defaults to "best" which is a tradeoff between accuracy and complexity.
    
    # Let's get the row corresponding to the selected equation
    # It takes some effort to match sympy back to row, but usually it is `model.get_best()`
    # but PySR API varies. `model.sympy()` gives the selected one.
    
    # Match the selected sympy expression back to its metrics row
    # In PySR, the selected equation (model.sympy()) corresponds to one row in model.equations_
    best_row = None
    for _, row in model.equations_.iterrows():
        # Compare sympy expressions for identity
        row_expr = sympy.sympify(row['equation'])
        if row_expr == best_eqn:
            best_row = row
            break
            
    if best_row is None:
        # Fallback to the one with max score if match fails
        best_row = model.equations_.loc[model.equations_['score'].idxmax()]
    
    return {
        "equation": eqn_str,
        "sympy_expr": best_eqn,
        "loss": float(best_row["loss"]),
        "score": float(best_row["score"]),
        "complexity": int(best_row["complexity"])
    }
