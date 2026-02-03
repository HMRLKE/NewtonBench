import networkx as nx
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Any, Union

class CausalEnv:
    def __init__(self, n_nodes: int = 5, seed: int = None, non_linear: bool = True):
        self.n_nodes = n_nodes
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.nodes = [f"Gene_{chr(65+i)}" for i in range(n_nodes)]
        self.dag = self._generate_random_dag()
        self.functions = self._assign_functions(non_linear)
        self.noise_std = 0.1
        
    def _generate_random_dag(self) -> nx.DiGraph:
        """Generates a random Directed Acyclic Graph."""
        # Start with empty graph
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        
        # Add random edges while maintaining acyclicity (forward edges in sorted list)
        # Shuffle order to ensure randomness in topology
        sorted_nodes = list(self.nodes)
        random.shuffle(sorted_nodes)
        
        # Dense-ish graph: p=0.3 edge prob
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if random.random() < 0.3:
                    u, v = sorted_nodes[i], sorted_nodes[j]
                    G.add_edge(u, v)
                    
        return G
        
    def _assign_functions(self, non_linear: bool) -> Dict[str, Any]:
        """Assigns causal mechanisms to each node."""
        funcs = {}
        for node in self.nodes:
            parents = list(self.dag.predecessors(node))
            
            if not parents:
                # Exogenous source
                funcs[node] = lambda p_vals: np.random.normal(0, 1)
            else:
                # Functional relationship
                if non_linear:
                    # Random non-linear mixture: sigmoid(sum(w*p)) or tanh or sin
                    weights = np.random.uniform(-1, 1, size=len(parents))
                    bias = np.random.uniform(-1, 1)
                    
                    # Closure to capture weights/bias
                    def mechanism(p_vals, w=weights, b=bias):
                        # p_vals is dict parent->val
                        x = sum(p_vals[p] * w_i for p, w_i in zip(parents, w)) + b
                        return np.tanh(x) # Biological-ish saturation
                    
                    funcs[node] = mechanism
                else:
                    # Linear
                    weights = np.random.uniform(-1, 1, size=len(parents))
                    bias = np.random.uniform(-0.5, 0.5)
                    
                    def mechanism_lin(p_vals, w=weights, b=bias):
                        return sum(p_vals[p] * w_i for p, w_i in zip(parents, w)) + b
                        
                    funcs[node] = mechanism_lin
        return funcs

    def simulate(self, n_samples: int = 100) -> pd.DataFrame:
        """Simulates observational data."""
        data = []
        for _ in range(n_samples):
            sample = self._sample_instance()
            data.append(sample)
        return pd.DataFrame(data)

    def _sample_instance(self, intervention: Dict[str, float] = None) -> Dict[str, float]:
        """Samples a single instance, respecting topological order."""
        values = {}
        # Intervention overrides
        intervention = intervention or {}
        
        # Topological traverse
        for node in nx.topological_sort(self.dag):
            if node in intervention:
                values[node] = intervention[node]
            else:
                # Get parent values
                parents = list(self.dag.predecessors(node))
                parent_vals = {p: values[p] for p in parents}
                
                # Compute mechanism + noise
                val = self.functions[node](parent_vals)
                val += np.random.normal(0, self.noise_std)
                values[node] = val
        return values

    def do_intervention(self, target_node: str, value: float, n_samples: int = 1) -> pd.DataFrame:
        """Performs a hard intervention do(target_node = value)."""
        if target_node not in self.nodes:
            raise ValueError(f"Node {target_node} not in graph")
            
        data = []
        for _ in range(n_samples):
            sample = self._sample_instance(intervention={target_node: value})
            data.append(sample)
        return pd.DataFrame(data)

    def get_ground_truth(self) -> nx.DiGraph:
        return self.dag

# Helper for run.py
import pandas as pd
