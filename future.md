# Feature Specification: Expanding AI Scientist Loop to Causal Discovery

## Context
Our current system is a "Physics Discovery Loop" implementing a closed-loop scientific workflow on NewtonBench tasks:
1.  **Experiment:** Interactive data collection.
2.  **Hypothesis:** Generates candidate laws via symbolic regression and constructs an *Equation Graph*.
3.  **Reviewer:** Validates using an LLM + deterministic metrics (MSE, stability, sanity checks).
4.  **Writer:** Generates a reproducible report based on the execution log.

## Objective
Extend the system to a new, non-physical domain (**Causal Discovery / Synthetic Biology**) to demonstrate generalization capabilities and strictly rule out *memorization*. The goal is to enable the **same non-fine-tuned (frozen) LLM** to uncover causal mechanisms in an unknown environment.

## New Domain: Synthetic Causal Graphs (The "Biology" Benchmark)
Instead of physical equations, the system must discover hidden Causal Graphs (Directed Acyclic Graphs - DAG).

### 1. Environment (The "Ground Truth")
*   **Generator:** Generate random synthetic causal networks at runtime (e.g., 5-15 nodes, random edges, non-linear transfer functions between nodes).
*   **Anti-memorization:** Since the graph structure is generated on the fly, it is impossible for the LLM to know the solution from its training data (unlike famous physical laws).
*   **Interface:** The environment operates as a black box.

### 2. Scientist Agent (Adaptation)
Instead of passive physical measurements, the scientist must perform *interventions*.
*   **Physics Mode Tool:** `measure(x_values)` -> returns `y_values`.
*   **Causal Mode Tool:** `do_intervention(node='Gene_A', value=0)` -> returns the new state of the system (values of all other nodes).
*   **Task:** Identify the topology of the graph. "Which variable causes which? Differentiate correlation from causation."

### 3. Hypothesis Representation
Generalize the current *Equation Graph* structure:
*   **Nodes:** Variables (e.g., Protein A, Protein B).
*   **Edges:** Directed relationships (A -> B).
*   **Format:** Adjacency Matrix or Edge List (JSON).

### 4. Reviewer & Metrics
Replace regression error metrics with graph similarity metrics:
*   **Metric:** SHD (Structural Hamming Distance) or F1-score (edges).
*   **Criteria:** How accurately was the hidden Ground Truth graph reconstructed? How many interventions were required (sample complexity)?

## Implementation Strategy (No Fine-tuning)
Adopt a "Universal Scientist Interface" pattern. Keep the architecture but swap the **Domain Context** in the prompt:

*   **Prompt Template:**
    > "You are an automated scientist. Your goal is to uncover the hidden mechanism of the environment."
    > [IF PHYSICS]: "Propose an equation that fits the measured data."
    > [IF CAUSAL]: "Propose a causal graph (DAG) that explains the intervention results. Use step-by-step reasoning to deduce directionality."

## Success Criteria
The system is considered successful if the LLM can correctly infer the topology of a randomly generated network through active experimentation (interventions) without any prior knowledge of the structure.