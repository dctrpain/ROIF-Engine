# ROIF Engine

> **Recursive Organic Integration Framework**
>
> Universal Cascade Analysis and Active Exploration Engine

---

## Overview

ROIF Engine is a domain-independent framework for analyzing, exploring, and reasoning about cascade dynamics in **pre-stressed complex systems**.

Unlike traditional graph analysis engines, ROIF is designed to operate on **incomplete causal graphs**.

Rather than assuming that the entire system is already known, ROIF actively identifies the most informative observation, proposes the next probe, reduces uncertainty, updates the graph, and only then performs causal inference.

Medicine serves as the first validation domain, not the architectural boundary.

---

# Why ROIF?

Most diagnostic systems answer the question:

> **"Given the graph, what is the cause?"**

ROIF asks a different question:

> **"What is the most informative next observation if the graph is incomplete?"**

This shift transforms ROIF from a passive analysis engine into an active exploration framework.

---

# Core Concepts

ROIF distinguishes four fundamentally different causal roles.

| Role | Meaning |
|-------|---------|
| **D_origin** | Structural origin of the cascade |
| **D_fast** | Earliest observable functional failure |
| **D_root** | Structural mediation node maintaining cascade propagation |
| **Node*** | Best intervention point |

These roles are intentionally independent.

---

# Active Probe Engine

One of the major architectural advances introduced in **v1.2.0** is the **Active Probe Engine (APE)**.

Instead of assuming a complete graph, ROIF performs active investigation.

```text
Incomplete Graph
        │
        ▼
Active Probe Engine
        │
        ▼
Probe*
        │
        ▼
Observation
        │
        ▼
Graph Update
        │
        ▼
Cascade Solver
        │
        ▼
D_origin
D_fast
D_root
Node*
```

The Active Probe Engine consists of:

- Probe Entities
- Probe Registry
- Probe Policy
- Probe Planner
- Probe Graph Adapter
- Active Probe Engine

---

# Cascade Solver

The Cascade Solver performs recursive propagation through pre-stressed systems.

Current capabilities include:

- Recursive cascade propagation
- Capacity tensor analysis
- Counterfactual simulation
- Mediation-based root detection
- Independent intervention ranking
- Structural history evaluation

---

# Counterfactual Engine

ROIF evaluates alternative intervention scenarios before recommending an action.

Supported analyses include:

- virtual restoration
- load reduction
- structural reinforcement
- outgoing influence modification
- incoming load reduction
- recursive scenario comparison

Counterfactual evaluation remains separated from causal inference.

---

# Validation

ROIF is developed using a validation-first approach.

Current validation includes:

- Unit tests
- Integration tests
- End-to-End tests
- Clinical validation
- Active Probe validation
- Counterfactual validation

Current regression suite:

> **6000+ automated tests**

---

# Repository Structure

```text
roif/
    active_probe_engine.py
    probe_entities.py
    probe_registry.py
    probe_policy.py
    probe_planner.py
    probe_graph_adapter.py

    solver.py
    network.py
    node.py
    element.py
    material.py

validation/
    clinical/

tests/

docs/
```

---

# Scientific Background

ROIF is developed alongside ongoing research into recursive cascade dynamics in pre-stressed systems.

The architecture supports concepts including:

- recursive cascade dynamics
- tensor-based propagation
- structural mediation
- counterfactual reasoning
- active exploration
- uncertainty reduction
- graph reconstruction

Current scientific terminology includes:

- D_origin
- D_fast
- D_root
- Node*
- Tensor W
- Spectral Coherence (η)

---

# Roadmap

Current development path:

```text
v1.0
    Cascade Engine

↓

v1.1
    Dynamic System Layer

↓

v1.2
    Active Probe Engine

↓

v1.3
    Graph Learning

↓

v1.4
    Recursive Active Exploration

↓

v1.5
    Knowledge Integration

↓

v2.0
    Recursive Active Inference Engine
```

See **ROADMAP.md** for the complete development plan.

---

# Current Status

Current Release:

**ROIF Engine v1.2.0**

Implemented:

- Stable Cascade Solver
- Counterfactual Engine
- Mediation-based D_root
- Independent Node*
- Active Probe Engine
- Clinical validation pipeline
- End-to-End validation
- 6000+ automated tests

---

# Design Principles

Every architectural decision should satisfy the following principles:

- Domain independent
- Mathematically formalizable
- Compatible with recursive cascade dynamics
- Compatible with Active Probe Engine
- Scientifically defensible
- Suitable for peer-reviewed publication

---

# License

See the LICENSE file for licensing information.