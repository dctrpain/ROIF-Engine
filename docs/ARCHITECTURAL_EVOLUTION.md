# ROIF Engine Architectural Evolution

> How ROIF evolved from a biomechanical simulation engine into a recursive active inference framework.

---

# Overview

ROIF Engine was not designed as a complete architecture from the beginning.

Instead, it evolved through a sequence of architectural transitions driven by practical limitations discovered during implementation, validation, and scientific development.

Each major version solved a specific class of problems while preserving compatibility with previous layers whenever possible.

This document explains why those architectural transitions occurred.

---

# Version 0.1 — XPBD Biomechanical Core

The first version of ROIF was a deterministic biomechanical simulation engine.

Its objective was to model adaptive pre-stressed biological structures using modern constraint-based mechanics.

Implemented components included:

- Node
- Element
- Material
- Network
- Solver
- XPBD constraints

The architecture successfully separated mechanics from constitutive behavior.

However, every computation remained essentially local.

The engine could simulate deformation, but not recursive causal propagation.

---

# Version 0.2 — Material Abstraction

The second stage introduced independent material models.

Instead of embedding constitutive equations inside mechanical elements, every material became a modular component.

Examples included:

- Muscle
- Tendon
- Ligament
- Fascia

This separation made it possible to extend the engine without modifying the mechanical core.

The architecture became significantly more modular.

---

# Version 0.3 — History-aware Mechanics

Clinical systems cannot be represented as memoryless mechanical systems.

This version introduced persistent state history.

New concepts included:

- fatigue;
- recovery;
- remodeling;
- rheological memory;
- irreversible structural adaptation.

The current state became dependent on previous system evolution rather than only the current load.

---

# Version 0.4 — Recursive Cascade Dynamics

As the project expanded beyond local mechanics, it became clear that many clinical phenomena could not be explained through isolated element behavior.

The architecture therefore shifted toward graph-based recursive propagation.

Major additions included:

- recursive influence propagation;
- tensor-based interactions;
- cascade simulation;
- network-wide state evolution.

ROIF gradually evolved from a mechanical simulator into a cascade engine.

---

# Version 0.5 — Counterfactual Engine

The next architectural step introduced virtual interventions.

Instead of analysing only the observed system, ROIF became capable of evaluating hypothetical scenarios without modifying the original graph.

The Counterfactual Engine introduced:

- virtual restoration;
- intervention comparison;
- recursive scenario simulation;
- intervention utility estimation.

This represented the first transition from passive observation toward decision support.

---

# Version 1.0 — Explicit Causal Roles

Experience showed that one numerical score could not describe every clinically relevant role.

The architecture therefore introduced four independent concepts:

- D_origin
- D_fast
- D_root
- Node*

Each role represents a different property of the cascade.

This separation removed many ambiguities that existed in earlier prototypes.

---

# Version 1.1 — Structural Mediation

Early implementations partially coupled D_root with intervention utility.

Clinical validation demonstrated that this produced incorrect behavior.

Version 1.1 redefined D_root as a structural mediator rather than the best intervention target.

The detector now evaluates characteristics such as:

- incoming influence;
- outgoing influence;
- directional balance;
- mediation throughput;
- downstream reach;
- tensor sensitivity.

Node* became fully independent and intervention-oriented.

This represented one of the most important architectural corrections in ROIF.

---

# Version 1.2 — Active Probe Engine

The next limitation concerned incomplete causal graphs.

Many real systems cannot be analysed reliably because important parts of the graph remain unknown.

Instead of ignoring uncertainty, ROIF introduced the Active Probe Engine (APE).

APE performs controlled exploration before definitive inference.

Major components include:

- Probe Registry;
- Probe Policy;
- Probe Planner;
- Active Probe Engine;
- Probe Graph Adapter.

Probe execution follows three phases:

```text
BASELINE

↓

PERTURBATION

↓

REASSESSMENT
```

Completed observations produce Graph Update Proposals instead of automatic graph mutations.

Explicit authorization is required before updating the graph.

This preserves auditability and safety.

---

# Clinical Validation

Version 1.2 introduced the first external validation package.

The Foot–Knee validation case demonstrated that:

- D_origin,
- D_fast,
- D_root,
- Node*

can be inferred independently while remaining clinically consistent.

Validation labels remain external references and are never supplied to the Solver.

---

# Current Architecture

ROIF Engine currently consists of several independent layers.

- Mechanical Core
- Material Layer
- Dynamic History
- Active Probe Engine
- Graph Adaptation
- Cascade Solver
- Counterfactual Engine
- Root Detection
- Validation
- Visualization

Each layer owns a clearly defined responsibility.

---

# Design Philosophy

Several principles guided the architectural evolution.

## Separation of Responsibilities

Every module should own one primary responsibility.

## Explicit Authorization

No high-risk modification should occur automatically.

## Domain Independence

Although medicine is the first validation domain, the architecture is intended for arbitrary pre-stressed complex systems.

## Recursive Thinking

Global behaviour emerges from recursive local interactions rather than centralized control.

## Scientific Reproducibility

Architectural decisions should remain mathematically formalizable and experimentally testable.

---

# Future Evolution

The next architectural milestone is ROIF Engine v2.0.

Planned capabilities include:

- graph learning;
- confidence propagation;
- adaptive graph reconstruction;
- recursive probe chains;
- uncertainty tensors;
- observation assimilation;
- recursive active inference.

The long-term objective is a framework capable of continuously improving its internal graph representation while preserving explicit safety and human oversight.

---

# Conclusion

ROIF has evolved from a deterministic biomechanical simulation engine into a recursive framework for causal analysis, counterfactual reasoning, and active exploration of incomplete graphs.

Future versions will extend these capabilities while preserving the architectural principles established during versions 0.1 through 1.2.