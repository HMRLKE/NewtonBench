import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict
import json

class CausalDiscoverer:
    """
    Agent that interacts with the CausalEnv to discover the graph structure.
    In a real full implementation, this would use an LLM or PC algorithm.
    For PoC, we implement a simple intervention-based logic or random guess to verify pipeline.
    """
    def __init__(self, env):
        self.env = env
        self.nodes = env.nodes
        self.history = []
        
    def discover(self):
        """
        Runs the discovery process.
        Returns the predicted graph (Adjacency List).
        """
        # 1. Observational Phase
        obs_data = self.env.simulate(n_samples=50)
        self.history.append({"action": "observe", "data_shape": obs_data.shape})
        
        # 2. Intervention Phase (Heuristic)
        # Try to perturb each node and see correlation changes?
        # For valid PoC without heavy LLM/Stats lib (like cdt), let's implementing a basic 
        # "perturb and check descendants" heuristic if feasible, or just return a dummy if we want to confirm pipeline first.
        
        # Let's try a simple correlation check on interventions to detect direction.
        # IF A causes B: do(A) changes B distribution. do(B) does NOT change A distribution.
        
        adj_matrix = {u: [] for u in self.nodes}
        
        # Baseline means
        baseline_means = obs_data.mean()
        
        for node in self.nodes:
            # Intervene on node (set to bias+2*std)
            val = baseline_means[node] + 2.0
            int_data = self.env.do_intervention(node, val, n_samples=30)
            int_means = int_data.mean()
            
            # Check which other nodes shifted significantly
            for other in self.nodes:
                if other == node: continue
                
                # Simple Z-test proxy: |mean_new - mean_base| > threshold
                # This assumes linear-ish response, but works for sign changes often.
                diff = abs(int_means[other] - baseline_means[other])
                if diff > 0.5: # Threshold
                    adj_matrix[node].append(other)
                    
            self.history.append({"action": f"do({node}={val:.2f})", "affected_nodes": adj_matrix[node]})
            
        return adj_matrix

def graph_to_json(adj_list):
    """Converts prediction to JSON format compatible with visualization."""
    G = nx.DiGraph()
    for u, targets in adj_list.items():
        for v in targets:
            G.add_edge(u, v)
            
    return {
        "nodes": [{"id": n} for n in G.nodes()],
        "links": [{"source": u, "target": v} for u, v in G.edges()]
    }
