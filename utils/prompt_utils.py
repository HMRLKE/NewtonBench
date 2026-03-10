import os
import json

def get_discovered_laws_context(accumulation_dir: str = "accumulation") -> str:
    """Read global_kg.json and return a formatted string of discovered laws."""
    kg_path = os.path.join(accumulation_dir, "global_kg.json")
    if not os.path.exists(kg_path):
        return ""
    
    try:
        with open(kg_path, "r") as f:
            data = json.load(f)
            
        laws = data.get("laws", [])
        if not laws:
            return ""
            
        context = "**Discovered Laws So Far:**\n"
        context += "Here are the symbolic laws that have been discovered in this universe for other phenomena. "
        context += "You may use these discovered laws as context to formulate consistent hypotheses about the underlying physical universe.\n"
        
        for law in laws:
            task = law.get("task", "Unknown Task")
            diff = law.get("difficulty", "unknown")
            eq = law.get("equation", "Unknown Equation")
            context += f"- Phenomenon: {task} (Difficulty: {diff})\n"
            context += f"  Discovered Equation: {eq}\n"
            
        return context
    except Exception as e:
        print(f"Error reading discovered laws: {e}")
        return ""
