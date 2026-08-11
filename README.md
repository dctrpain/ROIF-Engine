# ROIF Engine

> **Recursive Organic Integration Framework**
>
> Universal Cascade Analysis, Active Exploration, and Predictive Control Engine

---

## Overview

ROIF Engine is a domain-independent framework for analyzing, exploring, and controlling cascade dynamics in **pre-stressed complex systems**.

Unlike traditional graph analysis engines, ROIF is designed to operate on **incomplete causal graphs** and dynamically loaded systems.

Rather than assuming that the entire system is already known, ROIF can identify informative observations, perform controlled probes, evaluate directional and temporal responses, update the active cascade state, predict candidate control actions, and compare predicted outcomes with subsequent observations.

Medicine and biomechanics remain important validation domains, but they are not architectural boundaries.

ROIF is progressively being validated across mechanical and mixed human–physical systems.

---

# Why ROIF?

Most diagnostic or causal-analysis systems answer the question:

> **"Given the graph, what is the cause?"**

ROIF asks a broader sequence of questions:

> **"What is the most informative next observation if the graph is incomplete?"**

> **"Does the observed response actually support the proposed direction of propagation?"**

> **"What action is expected to stabilize the system under the current load and available reserve?"**

This transforms ROIF from a passive cascade-analysis engine into an architecture for **active exploration and predictive stabilization**.

---

# Core Causal Roles

ROIF distinguishes four fundamentally different causal and control roles.

| Role         | Meaning                                                   |
| ------------ | --------------------------------------------------------- |
| **D_origin** | Structural origin of the cascade                          |
| **D_fast**   | Earliest observable functional failure                    |
| **D_root**   | Structural mediation node maintaining cascade propagation |
| **Node***    | Best intervention point                                   |

These roles are intentionally independent.

A node may occupy more than one role in a particular system, but the architecture never assumes that the roles coincide.

---

# Cascade Engine

The Cascade Engine performs recursive propagation through pre-stressed systems.

Current capabilities include:

* recursive cascade propagation;
* Capacity Tensor analysis;
* vector-aware transport;
* utilization analysis;
* counterfactual simulation;
* mediation-based root detection;
* independent intervention ranking;
* structural history evaluation.

The passive cascade layer describes how disturbances can propagate through the currently represented system.

It does not by itself determine whether an observed active response confirms the causal direction being investigated.

---

# Active Probe Engine

The Active Probe Engine allows ROIF to investigate incomplete causal structure instead of assuming that the graph is already known.

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
Evidence
      │
      ▼
Authorized Graph / State Update
```

The Active Probe architecture includes:

* Probe Entities;
* Probe Registry;
* Probe Policy;
* Probe Planner;
* Probe Graph Adapter;
* Active Probe Engine;
* explicit authorization boundaries.

A proposed observation does not automatically become accepted structural knowledge.

---

# Vector Probe and Active Cascade

ROIF now separates passive transport from the response produced by an active probe.

This distinction is essential.

A branch may carry a large passive scalar response while producing an active response that is:

* aligned with the investigated direction;
* orthogonal to it;
* reversed;
* delayed;
* or insufficient.

The Vector Probe layer evaluates active evidence independently of passive amplitude.

Conceptually:

```text
Passive Cascade
      │
      ▼
Candidate Relations
      │
      ▼
Active Probe
      │
      ▼
Observed Response Vector
      │
      ▼
Directional / Temporal Evidence
      │
      ▼
Active Cascade Update
```

This prevents a high-amplitude response from automatically being interpreted as causal confirmation.

---

# Predictive Control

ROIF now contains a domain-independent **Predictive Control Layer**.

The layer evaluates the current system state, stabilization demand, available control reserve, and alternative candidate actions before selecting the next admissible control operation.

Conceptually:

```text
Current State
     +
Disturbance
     +
Control Reserve
     │
     ▼
Stabilization Demand
     │
     ▼
Candidate Actions
     │
     ▼
Predicted Outcomes
     │
     ▼
Candidate Ranking
     │
     ▼
ACTION / PROBE / HOLD / NO SAFE ACTION
```

The predictive controller does not require the strongest immediate action to be selected.

Candidate evaluation can account for:

* predicted residual error;
* stabilization demand;
* available reserve;
* uncertainty;
* action cost;
* reversibility;
* admissibility;
* probe value;
* safety constraints.

A system may remain temporarily stabilized while its available control reserve is already insufficient relative to stabilization demand.

ROIF therefore preserves **stabilization margin** as an explicit audit quantity rather than treating observed stability as proof of adequate reserve.

---

# Prediction Error

Predictive Control also introduces an explicit boundary between prediction and observation.

```text
Predicted State
      │
      ▼
Action / Probe
      │
      ▼
Observed State
      │
      ▼
Prediction Error
```

Prediction error records the difference between predicted and subsequently observed state variables.

This provides the architectural foundation for future experience-dependent adaptation.

Persistent controller memory and learning are **not yet implemented** in this layer.

---

# Counterfactual Engine

ROIF evaluates alternative intervention scenarios without modifying the original system.

Supported analyses include:

* virtual restoration;
* load reduction;
* structural reinforcement;
* outgoing influence modification;
* incoming load reduction;
* recursive scenario comparison.

Counterfactual evaluation remains separated from causal inference.

A useful intervention is not automatically the structural root of a cascade.

---

# Mixed-System Validation

ROIF validation has expanded beyond purely mechanical and clinical systems.

The **04A Sailing Yacht–Crew benchmark** represents a coupled system containing both:

### Mechanical Plant

* wind disturbance;
* sail load;
* heel/yaw response;
* course error;
* rudder action;
* sail-trim action.

### Living Controller

* helmsman control demand;
* sail-trimmer control demand.

The benchmark tests the architecture as a coupled **physical plant + living controller** system.

The predictive-control layer evaluates several candidate stabilization strategies, including helm correction, sail-trim correction, coupled helm–trim action, holding the current control state, and an information probe.

This benchmark is intended as a cross-domain validation environment for control architecture, not as a claim that human cognition has already been fully modeled.

---

# Validation

ROIF is developed using a validation-first approach.

Current validation includes:

* Unit tests;
* Integration tests;
* End-to-End tests;
* Clinical validation;
* Mechanical ground-truth validation;
* Active Probe validation;
* Active Cascade validation;
* Vector-response validation;
* Temporal-response validation;
* Counterfactual validation;
* Predictive Control validation;
* Mixed yacht–crew validation.

Important mechanical controls include cases in which:

* the largest-amplitude branch is directionally wrong;
* a high-amplitude response is orthogonal to the candidate direction;
* a high-amplitude response is reversed;
* the correct response is delayed;
* early observation would produce a misleading conclusion.

These controls explicitly test whether ROIF can avoid treating **amplitude, timing, or passive transport as sufficient causal evidence**.

---

# Current Validation Benchmarks

Current validation families include:

```text
Mechanical
├── Pre-stressed spring
├── Serial weak link
├── Pre-stressed branching
├── Symmetric branching ambiguity
├── 03C Misleading High-Amplitude Branch
├── 03D Misleading Reverse-Direction Response
└── 03E Delayed Misleading Response

Clinical / Biomechanical
├── Foot–Knee validation
├── Active Probe validation
└── Recursive Probe validation

Mixed Human–Physical Systems
└── 04A Sailing Yacht–Crew
    ├── coupled-system ground truth
    └── predictive stabilization
```

---

# Repository Structure

```text
roif/
    active_probe_engine.py
    active_cascade.py
    active_cascade_ape.py
    vector_probe.py
    utilization.py
    predictive_control.py

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
    mechanical/
    sailing/

tests/

docs/
```

---

# Current Architecture

The current ROIF architecture can be represented as:

```text
Pre-Stressed System
        │
        ▼
Passive Cascade
        │
        ▼
Incomplete / Candidate Structure
        │
        ▼
Active Probe
        │
        ▼
Vector + Temporal Evidence
        │
        ▼
Active Cascade
        │
        ▼
Current Dynamic State
        │
        ▼
Predictive Control
        │
        ├── Stabilization Demand
        ├── Control Reserve
        ├── Candidate Actions
        └── Predicted Outcomes
        │
        ▼
Action / Probe / Hold
        │
        ▼
Observation
        │
        ▼
Prediction Error
```

The architecture intentionally keeps these layers separate.

---

# Two Different Forms of History

ROIF already contains physical and structural history mechanisms.

Examples include:

* fatigue;
* recovery;
* remodeling;
* rheological memory;
* irreversible material adaptation.

These mechanisms describe how **the physical system itself changes because of its previous loading history**.

A future research layer will investigate a different form of history:

```text
Prediction
    ↓
Action
    ↓
Outcome
    ↓
Prediction Error
    ↓
Experience Trace
```

Such experience-dependent controller memory must remain conceptually separate from rheological or structural material memory.

It is **not yet an implemented capability**.

---

# Scientific Background

ROIF is developed alongside ongoing research into recursive cascade dynamics and adaptive control in pre-stressed systems.

The architecture currently supports concepts including:

* recursive cascade dynamics;
* tensor-based propagation;
* structural mediation;
* counterfactual reasoning;
* active exploration;
* vector-sensitive evidence;
* temporal evidence;
* uncertainty reduction;
* predictive stabilization;
* finite control reserve;
* prediction error.

Current scientific terminology includes:

* D_origin;
* D_fast;
* D_root;
* Node*;
* Probe*;
* Tensor W;
* Spectral Coherence (η);
* Stabilization Demand;
* Control Reserve;
* Prediction Error.

---

# Current Status

Implemented:

* Stable Cascade Solver;
* Pre-stressed system mechanics;
* Capacity Tensor;
* Dynamic History;
* Rheological Memory;
* Counterfactual Engine;
* Mediation-based D_root;
* Independent Node*;
* Active Probe Engine;
* Vector Probe;
* Utilization layer;
* Active Cascade;
* Direction-sensitive active evidence;
* Temporal-response controls;
* Predictive Control;
* Prediction Error comparison;
* Clinical validation pipeline;
* Mechanical validation suite;
* 04A Yacht–Crew coupled-system benchmark;
* 04A Yacht–Crew predictive-control benchmark.

---

# Next Research Layer

The next architectural research step is **experience-dependent predictive control**.

The proposed sequence is:

```text
Prediction
    ↓
Action / Probe
    ↓
Observed State
    ↓
Prediction Error
    ↓
Experience Trace
    ↓
Persistent Controller Memory
    ↓
Modified Predictive State
    ↓
Next Prediction
```

Working concepts such as **Memory Scar** and **Predictive Preload** remain research terminology until they are formally defined, implemented, and validated.

The immediate implementation target is therefore the smallest auditable representation of an **Experience Trace**.

---

# Cross-Domain Development Direction

```text
Mechanical Validation
        │
        ▼
Active Probe Controls
        │
        ▼
Vector / Temporal Evidence
        │
        ▼
Mixed Yacht–Crew System
        │
        ▼
Predictive Stabilization
        │
        ▼
Experience-Dependent Control
        │
        ▼
Adaptive Stabilization
        │
        ▼
Robotics Transfer Validation
```

The purpose of cross-domain validation is to test whether the same core architecture remains reusable without redesigning its fundamental inference and control principles.

---

# Design Principles

Every architectural decision should satisfy the following principles:

* Domain independent;
* Mathematically formalizable;
* Compatible with recursive cascade dynamics;
* Compatible with pre-stressed systems;
* Compatible with Active Probe;
* Explicit about uncertainty;
* Explicit about control reserve;
* Auditable;
* Deterministic where required by validation;
* Scientifically defensible;
* Experimentally testable;
* Suitable for peer-reviewed publication.

---

# Architectural Safety

ROIF preserves explicit authorization boundaries between evidence and structural modification.

The **Non-Fonit Gate** remains a hard architectural constraint.

Information gain, predictive utility, or expected stabilization benefit must not override unacceptable cascade risk.

---

# Research Boundary

ROIF currently implements predictive stabilization and active exploration.

It does **not** currently claim to implement:

* consciousness;
* biological instinct;
* human cognition;
* autonomous psychological learning;
* a complete model of biological predictive processing.

Possible relationships between repeated prediction errors, persistent controller memory, learned stabilization responses, and biological control remain **research hypotheses to be formalized and tested**.

---

# License

See the LICENSE file for licensing information.
