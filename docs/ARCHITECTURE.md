# ROIF Engine Architecture

> Recursive Organic Integration Framework  
> Architecture specification for ROIF Engine v1.2.0

## Overview

ROIF Engine is a domain-independent framework for simulation, cascade analysis, counterfactual evaluation, and active exploration of incomplete pre-stressed complex systems.

Medicine and biomechanics are the first validation domains, not architectural boundaries.

The architecture emphasizes:

- modularity;
- deterministic simulation;
- scientific reproducibility;
- separation of mechanics from material behavior;
- separation of inference from intervention;
- explicit handling of incomplete graphs;
- auditable safety and authorization boundaries.

---

## High-Level Architecture

```text
External System
      |
      v
Graph Representation
      |
      +---------------------------+
      |                           |
      v                           v
Known Graph                Incomplete Graph
      |                           |
      |                           v
      |                IncompleteGraphSnapshot
      |                           |
      |                           v
      |                  Probe Graph Adapter
      |                           |
      |                           v
      |              ProbeInformationEstimate
      |                           |
      |                           v
      |                Active Probe Engine
      |                           |
      |                           v
      |                        Probe*
      |                           |
      |                           v
      |         BASELINE -> PERTURBATION
      |                           |
      |                           v
      |                    REASSESSMENT
      |                           |
      |                           v
      |                     ProbeResult
      |                           |
      |                           v
      |                GraphUpdateProposal
      |                           |
      |                           v
      |                Explicit Authorization
      |                           |
      +-------------+-------------+
                    |
                    v
             Authorized Graph
                    |
                    v
             Cascade Solver
                    |
          D_origin / D_fast / D_root / Node*
```

A `GraphUpdateProposal` is not an automatic graph mutation.

---

## Core Mechanical Layer

### Node

Stores local state:

- position;
- velocity;
- accumulated forces;
- mass;
- fixed/free state.

Nodes contain no constitutive material behavior.

### Element

Represents an interaction between nodes.

Responsibilities:

- compute current length;
- compute extension and strain;
- request material forces;
- apply forces to connected nodes.

### Material

Defines constitutive behavior independently from topology.

Responsibilities may include:

- elasticity;
- damping;
- fatigue;
- recovery;
- remodeling;
- failure;
- history-dependent behavior.

Current models include muscle, tendon, ligament, fascia, and generic materials.

### Network

Owns the global system state.

Responsibilities:

- node and element management;
- topology management;
- force accumulation;
- simulation stepping;
- solver interaction;
- state snapshots.

### Solver

Responsible for numerical evolution.

Current capabilities include:

- explicit integration;
- XPBD constraints;
- deterministic stepping;
- recursive cascade propagation;
- history-aware computation.

---

## Dynamic History Layer

ROIF does not treat the system as memoryless.

The history layer supports:

- rheological memory;
- fatigue accumulation;
- recovery;
- remodeling;
- persistent geometry changes;
- structural adaptation.

---

## Active Probe Engine

The Active Probe Engine (APE) was introduced in v1.2.0.

Its purpose is to reduce graph uncertainty before causal inference.

A Probe is a controlled experiment:

```text
Perturbation
      |
      v
Measurement
      |
      v
Information about the graph
```

Each Probe follows:

```text
BASELINE -> PERTURBATION -> REASSESSMENT
```

The difference between baseline and reassessment is represented by `ObservationDelta`.

APE components:

- `probe_entities.py`;
- `probe_registry.py`;
- `probe_policy.py`;
- `probe_planner.py`;
- `active_probe_engine.py`;
- `probe_graph_adapter.py`.

### Probe Registry

Stores reusable `ProbeDefinition` objects.

It handles registration, lookup, filtering, uniqueness, and immutable snapshots.

### Probe Policy

Evaluates admissibility.

```text
REJECT > REQUIRE_AUTHORIZATION > ALLOW
```

Policy may evaluate:

- cost and duration;
- method and perturbation restrictions;
- risk and cascade risk;
- uncertainty;
- reversibility;
- resources;
- human authorization.

### Non-Fonit Gate

A mandatory veto rejects Probes capable of uncontrolled, external, or large-scale cascade effects.

Human authorization does not automatically override this veto.

### Probe Planner

Ranks admissible candidates using:

- information gain;
- uncertainty reduction;
- hypothesis discrimination;
- graph coverage;
- novelty;
- feasibility;
- confidence;
- cost;
- duration;
- risk;
- redundancy;
- disruption.

The Planner selects `Probe*`, but does not execute it.

### Probe Lifecycle

```text
IDLE -> PLANNED -> RUNNING -> COMPLETED / CANCELLED
```

The Active Probe Engine stores observations, derives deltas, and creates `ProbeResult`.

It does not physically apply perturbations, mutate the graph, or call the Cascade Solver.

---

## Graph Adaptation Layer

`probe_graph_adapter.py` performs:

```text
IncompleteGraphSnapshot -> ProbeInformationEstimate
```

and:

```text
ProbeResult -> GraphUpdateProposal
```

The adapter never applies the proposal automatically.

```text
GraphUpdateProposal != Graph Mutation
```

---

## Causal Roles

ROIF separates four roles.

### D_origin

Structural origin of the cascade.

### D_fast

Earliest channel that loses functional support or reserve.

### D_root

Principal structural mediator of cascade propagation.

D_root must both receive upstream influence and transmit influence downstream.

The v1.2.0 detector uses mediation characteristics such as incoming strength, outgoing strength, balance, throughput, downstream reach, and tensor sensitivity.

D_root is not defined by intervention utility.

### Node*

Optimal intervention point under current constraints.

Its evaluation may include gain, cost, collateral effects, uncertainty, safety, reversibility, and confidence.

```text
D_origin != D_fast != D_root != Node*
```

Roles may coincide in a particular system, but coincidence is never assumed.

---

## Counterfactual Engine

The Counterfactual Engine evaluates hypothetical interventions without mutating the original system.

Supported concepts include:

- virtual restoration;
- outgoing influence modification;
- incoming load reduction;
- structural reinforcement;
- scenario comparison;
- recursive cascade simulation.

Counterfactual utility belongs to `Node*` evaluation and must not define `D_root`.

---

## Validation Layer

Validation includes:

- unit tests;
- integration tests;
- end-to-end tests;
- role-separation tests;
- Active Probe tests;
- graph-adapter tests;
- clinical validation.

The first clinical validation package is located in:

```text
validation/clinical/
```

External validation labels must not be supplied to the Solver as hidden answers.

---

## Safety Boundaries

ROIF preserves:

- explicit authorization;
- preference for reversible Probes;
- cascade-risk evaluation;
- Non-Fonit veto;
- proposal-only graph updates;
- separation of inference and intervention;
- auditability.

In clinical use, the physician remains the external decision authority.

---

## Current Release

**ROIF Engine v1.2.0**

Implemented:

- stable mechanical and cascade solver;
- material and constraint architecture;
- dynamic history and rheological memory;
- Counterfactual Engine;
- mediation-based D_root;
- independent Node*;
- Active Probe Engine;
- incomplete graph adapter;
- clinical validation pipeline;
- more than 6000 automated tests.

---

## Future Evolution

The next stage is Graph Learning:

- observation assimilation;
- confidence propagation;
- incremental graph updates;
- graph uncertainty tensor;
- Probe history;
- multi-step Probe planning;
- adaptive graph reconstruction.

The long-term target is ROIF Engine v2.0: a Recursive Active Inference Engine.

---

## Guiding Principle

ROIF separates structure, mechanics, material behavior, active exploration, graph adaptation, causal inference, and intervention planning.

Complex behavior should emerge from modular interactions rather than being hard-coded into a monolithic system.
