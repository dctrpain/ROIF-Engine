# ROIF Engine Architecture

> Recursive Organic Integration Framework
>
> Architecture specification for ROIF Engine v1.4.0

**Current release:** `v1.4.0`
**Release date:** 2026-08-20
**Release commit:** `580716c13fd4866484cbb651157002123e337b06`

---

## 1. Overview

ROIF Engine is a domain-independent computational architecture for representing, simulating, probing, evolving, and experimentally separating response mechanisms in incomplete pre-stressed complex systems.

Medicine and biomechanics are validation domains, not architectural boundaries.

The architecture emphasizes:

- modularity;
- deterministic and reproducible computation;
- separation of mechanics from constitutive material behavior;
- separation of physical state from retained structured memory;
- separation of passive response from active causal evidence;
- separation of prediction from observation;
- separation of inference from intervention;
- explicit handling of incomplete graphs;
- explicit causal-role separation;
- finite control reserve;
- provenance-preserving transition conditioning;
- auditable safety and authorization boundaries;
- explicit scientific claim boundaries.

ROIF is not architecturally defined as a scalar time-series forecasting system. Its temporal components operate on represented system organization, state, coupling, memory, and transition mechanisms under explicitly defined experimental conditions.

---

## 2. High-Level Architecture

```text
External / Represented System
            |
            v
      Graph + System State
            |
      +-----+--------------------------------------+
      |                                            |
      v                                            v
Known / Incomplete Graph                    Physical-State Path
      |                                            |
      v                                            v
Active Probe / Cascade                       SystemEvolution
      |                                            |
      |                                  +---------+---------+
      |                                  |                   |
      |                                  v                   v
      |                           Current Physical      Structured
      |                              Organization         Memory
      |                                                      |
      |                                                      v
      |                                        Memory-to-Transition
      |                                             Derivation
      |                                                      |
      |                                                      v
      |                                           TransitionModifierSet
      |                                                      |
      |                                                      v
      |                                           PRESTRESS_TRANSFER
      |                                                      |
      +-----------------------------+------------------------+
                                    |
                                    v
                              Next System State
                                    |
             +----------------------+----------------------+
             |                                             |
             v                                             v
      Observation / Evidence                         Predictive Control
             |                                             |
             v                                             v
       Graph / State Update                         Predicted Outcome
                                                           |
                                                           v
                                                      Observation
                                                           |
                                                           v
                                                    Prediction Error
```

This diagram combines several architectural paths, but they must not be interpreted as one undifferentiated mechanism.

In particular:

```text
Physical History != Structured Retained Memory

Passive Response != Active Causal Evidence

Prediction != Observation

TransitionModifierSet != General History-Conditioned Operator

Temporal Reconstruction != General Future Forecasting
```

---

## 3. Core Mechanical Layer

### 3.1 Node

A `Node` stores local mechanical state such as:

- position;
- velocity;
- accumulated forces;
- mass;
- fixed/free state.

Nodes do not own constitutive material behavior.

### 3.2 Element

An `Element` represents an interaction between nodes.

Responsibilities include:

- current-length computation;
- extension and strain computation;
- interaction with constitutive material models;
- force application to connected nodes.

### 3.3 Material

Material models define constitutive behavior independently from graph topology.

Capabilities may include:

- elasticity;
- damping;
- fatigue;
- recovery;
- remodeling;
- failure;
- history-dependent constitutive behavior.

Implemented material families include muscle, tendon, ligament, fascia, and generic materials.

### 3.4 Network

The `Network` owns global mechanical organization.

Responsibilities include:

- node and element management;
- topology management;
- force accumulation;
- simulation stepping;
- solver interaction;
- state snapshots.

### 3.5 Solver

The solver provides numerical evolution.

The implemented architecture includes:

- explicit integration;
- XPBD constraints;
- deterministic stepping;
- recursive cascade computation;
- history-aware mechanical computation.

---

## 4. Pre-Stressed Cascade Layer

ROIF represents systems in which local perturbation can redistribute load through an already stressed network.

The cascade layer supports:

- graph-based propagation;
- recursive propagation;
- direction-aware transport;
- Capacity Tensor calculations;
- utilization analysis;
- downstream redistribution;
- state-dependent response.

A large response at one node does not automatically identify the structural origin, structural mediator, optimal intervention point, or best information-gathering Probe.

This motivates explicit causal-role separation.

---

## 5. Dynamic Physical History

ROIF does not assume that the present physical system is memoryless.

Physical history may be embodied in present organization through mechanisms such as:

- fatigue accumulation;
- recovery;
- remodeling;
- rheological memory;
- persistent geometry change;
- structural adaptation;
- adaptive connection state;
- prestress redistribution.

Conceptually:

```text
Past Loading
     |
     v
Changed Physical Organization
     |
     v
Changed Present Physical State
     |
     v
Changed Subsequent Response
```

This is **state-mediated history dependence**.

It is not equivalent to structured memory retained outside the matched present physical state.

---

## 6. Recursive System Evolution

`SystemEvolution` provides an executable path for recursive physical-state transition.

Conceptually:

```text
SystemImage(t)
      |
      v
Transition Inputs
      |
      v
SystemEvolution
      |
      v
SystemImage(t+1)
```

The evolution path may include changes in physical organization, adaptive state, and supported transition channels.

Release `v1.4.0` additionally allows explicitly derived transition modifiers to condition the supported `PRESTRESS_TRANSFER` channel.

The legacy evolution path is preserved when transition modifiers are absent or zero.

---

## 7. Structured Memory Layer

Release `v1.4.0` introduces an explicit distinction between:

1. history embodied in current physical organization; and
2. structured retained memory that can remain distinguishable from a matched current physical state.

Structured memory is not allowed to alter physical evolution implicitly.

Instead, the architecture requires an explicit derivation boundary:

```text
Structured Memory
       |
       v
Memory-to-Transition Derivation
       |
       v
TransitionModifierSet
       |
       v
Authorized Physical Transition Channel
```

This preserves inspectability between semantic historical information and executable physical transition modification.

---

## 8. Memory-to-Transition Derivation

`roif/history/memory_transition_derivation.py` implements deterministic derivation of bounded transition modifiers from structured memory evidence.

The derivation layer provides:

- explicit semantic-to-physical target bindings;
- binding polarity;
- target-channel mapping;
- deterministic derivation;
- provenance preservation;
- rejection of ambiguous duplicate physical transition targets.

The derivation layer does not itself evolve the physical system.

Its role is to translate authorized structured memory into an explicit bounded transition-conditioning representation.

---

## 9. TransitionModifierSet

`roif/history/transition_modifiers.py` defines the transition-conditioning interface.

The architecture provides:

- transition channels;
- bounded modifier values;
- deterministic modifier signatures;
- structured provenance;
- explicit target identity;
- a policy-free modifier representation.

Conceptually:

```text
Memory Evidence
      |
      v
Derivation
      |
      v
TransitionModifierSet
      |
      v
SystemEvolution
```

### 9.1 Current Executable Channel

In `v1.4.0`, executable integration is intentionally restricted to:

```text
PRESTRESS_TRANSFER
```

This channel allows bounded modification of effective prestress-transfer behavior during a subsequent transition.

Unsupported active modifier channels are rejected explicitly rather than being silently interpreted.

### 9.2 Zero-Modifier Invariant

If transition modifiers are absent or zero, the legacy system-evolution path is preserved exactly.

This is an important compatibility and experimental-control invariant.

### 9.3 Scientific Boundary

```text
TransitionModifierSet
        !=
General History-Conditioned Operator
over the complete SystemImage
```

The current implementation demonstrates a bounded memory-conditioned transition mechanism through the tested physical channel.

It does not establish unrestricted history-conditioned evolution of the complete system representation.

---

## 10. Matched-Current-State History Separation

Release `v1.4.0` contains experimental controls designed to distinguish state-mediated physical history from retained structured history.

Conceptually:

```text
History A -> Physical State A
History B -> Physical State B
               |
               v
        Matched Current State
          /             \
         /               \
Memory A                 Memory B
   |                        |
   v                        v
Modifier Set A          Modifier Set B
         \               /
          \             /
           v           v
       Subsequent Transition
```

The experimental question is whether retained structured history can remain identifiable after relevant current physical state has been matched.

This is a mechanism-specific identifiability test.

It is not evidence for a universal history operator.

---

## 11. Active Probe Engine

The Active Probe Engine (APE) provides controlled exploration of incomplete graphs.

A Probe is represented as an information-generating intervention:

```text
BASELINE
    |
    v
PERTURBATION
    |
    v
REASSESSMENT
    |
    v
ObservationDelta
    |
    v
ProbeResult
```

Core components include:

- `probe_entities.py`;
- `probe_registry.py`;
- `probe_policy.py`;
- `probe_planner.py`;
- `active_probe_engine.py`;
- `probe_graph_adapter.py`.

### 11.1 Probe Registry

Stores reusable `ProbeDefinition` objects and supports:

- registration;
- lookup;
- filtering;
- uniqueness;
- immutable snapshots.

### 11.2 Probe Policy

Evaluates admissibility.

```text
REJECT > REQUIRE_AUTHORIZATION > ALLOW
```

Policy may evaluate:

- cost;
- duration;
- perturbation restrictions;
- cascade risk;
- uncertainty;
- reversibility;
- resources;
- human authorization.

### 11.3 Probe Planner

Ranks admissible candidates using criteria such as:

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

The planner selects `Probe*`.

It does not physically execute the Probe.

### 11.4 Probe Lifecycle

```text
IDLE -> PLANNED -> RUNNING -> COMPLETED / CANCELLED
```

The Active Probe Engine records observations, derives deltas, and creates `ProbeResult`.

It does not silently mutate the graph.

---

## 12. Graph Adaptation Boundary

`probe_graph_adapter.py` supports:

```text
IncompleteGraphSnapshot
        |
        v
ProbeInformationEstimate
```

and:

```text
ProbeResult
     |
     v
GraphUpdateProposal
```

A proposal is not automatically accepted as system structure.

Architectural invariant:

```text
GraphUpdateProposal != Graph Mutation
```

Explicit authorization remains a separate boundary.

---

## 13. Vector Probe Evidence

ROIF separates scalar response amplitude from directional evidence.

The Vector Probe layer evaluates whether an observed response is directionally compatible with a tested relation.

Evidence outcomes include:

```text
SUPPORTS
CONTRADICTS
INSUFFICIENT
NO_RESPONSE
```

Architectural invariant:

```text
Large Passive Response != Directional Causal Confirmation
```

A high-amplitude response can therefore fail to support a hypothesized relation if its directional structure is incompatible.

---

## 14. Utilization-Aware Evidence

ROIF can incorporate utilization change into active evidence evaluation.

This provides an additional distinction between:

- absolute response;
- directional response;
- change in system utilization.

Utilization evidence complements rather than replaces structural, directional, and temporal evidence.

---

## 15. Temporal Active Evidence

Probe interpretation can depend on an observation window.

The architecture supports distinctions between:

- early observation;
- delayed response;
- full observation window;
- insufficient temporal evidence.

This prevents a relation from being authorized solely because it produced the first visible response.

Architectural invariant:

```text
First Visible Response != Final Causal Decision
```

Temporal evidence in this layer concerns observation of active perturbation response.

It must not be confused with the separate Temporal Image reconstruction benchmarks.

---

## 16. Active Cascade

The Active Cascade layer allows recursive progression through actively supported relations.

It maintains explicit records of:

- confirmed edges;
- rejected relations;
- current-node transition state;
- evidence provenance;
- audit information.

Conceptually:

```text
Probe
  |
  v
Evidence
  |
  v
Relation Decision
  |
  v
Authorized Active Transition
  |
  v
Next Candidate Relation
```

Active Cascade therefore differs from passive propagation through a predefined graph.

---

## 17. Explicit Causal Roles

ROIF separates five roles.

### 17.1 D_origin

Structural origin of the cascade.

### 17.2 D_fast

Earliest channel that loses functional support or reserve.

### 17.3 D_root

Principal structural mediator of cascade propagation.

`D_root` must be evaluated as a propagation role, not as intervention utility.

### 17.4 Node*

Counterfactual intervention candidate under current constraints.

Evaluation may include:

- expected gain;
- cost;
- collateral effects;
- uncertainty;
- safety;
- reversibility;
- confidence.

### 17.5 Probe*

Information-oriented candidate selected to reduce relevant uncertainty.

The central invariant is:

```text
D_origin != D_fast != D_root != Node* != Probe*
```

These roles may coincide in a particular system, but coincidence is never assumed by the architecture.

---

## 18. Counterfactual Engine

The Counterfactual Engine evaluates hypothetical interventions without mutating the original system.

Supported concepts include:

- virtual restoration;
- outgoing influence modification;
- incoming-load reduction;
- structural reinforcement;
- scenario comparison;
- recursive cascade simulation;
- intervention utility estimation.

Counterfactual utility belongs to `Node*` evaluation.

It must not define `D_root`.

Architectural invariant:

```text
Useful Intervention != Structural Root
```

---

## 19. Predictive Control

ROIF includes a domain-independent predictive-control layer.

Implemented representations include:

- `PredictiveState`;
- `DisturbanceEstimate`;
- `ControlReserve`;
- `StabilizationDemand`;
- `ControlCandidate`;
- `PredictedOutcome`;
- `CandidateEvaluation`;
- `PredictiveControlDecision`;
- `PredictionError`.

The decision space includes:

```text
ACTION
PROBE
HOLD
NO_SAFE_ACTION
```

The controller can:

- estimate stabilization demand;
- represent finite control reserve;
- evaluate candidate actions;
- rank predicted outcomes;
- select a reversible Probe when action is insufficiently justified;
- compare prediction with subsequent observation.

A key invariant is:

```text
Observed Stability != Adequate Control Reserve
```

A system may remain apparently stable while its remaining stabilization margin declines.

---

## 20. Prediction-Observation Boundary

ROIF explicitly preserves the distinction between predicted and observed outcomes.

```text
Current State
      |
      v
Prediction
      |
      v
Action / Probe
      |
      v
Observation
      |
      v
Prediction Error
```

`PredictionError` is an auditable comparison signal.

It is not, by itself, persistent learning.

Architectural invariant:

```text
Prediction Error != Learning
```

---

## 21. Predictive Preconfiguration

Release `v1.4.0` contains a restricted one-step predictive-preconfiguration benchmark.

The implemented mechanism tests whether defined prestress preconfiguration can improve a specified objective under controlled disturbance and finite reserve.

The associated validation includes:

- predictive-preconfiguration benchmark;
- objective-independence audit;
- off-nominal transferability audit;
- claim-boundary tests.

### Scientific Boundary

The implemented benchmark does not establish:

- objective-independent optimal control;
- general whole-system predictive stabilization;
- unrestricted future-state forecasting;
- learned anticipation;
- biological prediction;
- universal stability.

Architectural interpretation:

```text
Restricted Predictive Preconfiguration
        !=
General Predictive Stabilization
```

---

## 22. Temporal Image Experimental Layer

ROIF includes Temporal Image experiments that test explicitly separated temporal properties.

Current assets include:

- Temporal Image reconstruction benchmark;
- reconstruction-method comparison;
- reconstruction slices;
- multilayer temporal-image trajectory benchmark;
- fixed-coupling controls;
- memory-free controls.

The temporal benchmark must not be interpreted as a single competition between ROIF and linear interpolation.

Different conditions test different architectural properties.

### Anti-Linear Reading Protocol

The Temporal Image program is not architecturally defined as ordinary scalar time-series forecasting.

The relevant question is not merely:

```text
What scalar value comes next?
```

The program instead tests how represented system state, coupling, retained history, and trajectory information affect reconstruction or transition behavior under controlled conditions.

Therefore:

```text
Temporal Image Reconstruction
        !=
Complete Future Temporal Image Prediction
```

and:

```text
Trajectory Dependence
        !=
General Forecasting Capability
```

---

## 23. History-Conditioned Redistribution Experiments

Release `v1.4.0` includes controls for history-conditioned redistribution.

These experiments examine contributions associated with mechanisms such as:

- prestress organization;
- adaptive connections;
- represented historical state;
- matched-current-state controls.

Their purpose is to determine which represented mechanisms alter subsequent redistribution under defined conditions.

They do not establish a universal memory operator.

---

## 24. Mixed Physical-Living Controller Validation

The sailing yacht-crew validation environment tests transfer to a coupled physical plant and living controller.

Conceptually:

```text
Environment
    |
    v
Physical Plant
    |
    v
System Error
    |
    v
Living Controller
    |
    v
Control Action
    |
    v
Physical Plant
```

The benchmark is not intended as a complete model of human cognition.

Its architectural purpose is to test separation between:

- physical dynamics;
- system error;
- finite control reserve;
- control action;
- predictive evaluation;
- resulting physical response.

---

## 25. Validation and Reproducibility Layer

Validation includes:

- unit tests;
- integration tests;
- end-to-end tests;
- causal-role separation tests;
- Active Probe tests;
- graph-adapter tests;
- Vector Probe tests;
- temporal-evidence tests;
- Active Cascade tests;
- predictive-control tests;
- SystemEvolution tests;
- transition-modifier tests;
- memory-transition derivation tests;
- matched-state benchmark tests;
- predictive-preconfiguration tests;
- Temporal Image benchmark tests;
- architecture claim-boundary tests.

Release `v1.4.0` also includes manuscript-facing:

- benchmark scripts;
- benchmark result artifacts;
- figure-generation code;
- explicit claim-boundary tests.

Full repository regression checkpoint:

```text
10877 passed
```

Release checkpoint:

```text
v1.4.0
580716c13fd4866484cbb651157002123e337b06
```

This checkpoint is the reproducibility reference for the current release.

---

## 26. Safety and Authorization Boundaries

ROIF preserves:

- explicit authorization;
- preference for reversible Probes;
- cascade-risk evaluation;
- proposal-only graph updates;
- separation of inference from intervention;
- provenance;
- auditability;
- finite control constraints;
- Non-Fonit veto.

### Non-Fonit Gate

The Non-Fonit Gate is a hard safety veto against uncontrolled, external, or large-scale cascade risk.

Conceptually:

```text
Information Gain
Predictive Benefit
Control Utility
        |
        v
cannot override
        |
        v
Unacceptable Cascade Risk
```

Human authorization does not automatically override the veto.

No future prediction, learning, optimization, graph-adaptation, or active-exploration layer may bypass this boundary.

In clinical use, the clinician remains the external decision authority.

---

## 27. Architectural Separation of Memory Mechanisms

The word "memory" can refer to different mechanisms and must not be used as if they were interchangeable.

### 27.1 Physical / Rheological Memory

Embodied in current physical organization.

Examples include:

- fatigue;
- remodeling;
- changed geometry;
- adaptive connections;
- prestress organization.

### 27.2 Structured Retained Memory

Represented outside the matched current physical state and capable, through explicit derivation, of producing bounded transition modifiers.

### 27.3 Future Controller Experience

A future controller-memory layer may retain prediction-action-observation experience for later candidate evaluation.

This is not yet equivalent to the implemented structured memory-to-transition path.

Architectural invariant:

```text
Physical Memory
    !=
Structured Retained Memory
    !=
Future Controller Memory
```

---

## 28. Graph Learning and Controller Learning Are Separate

Future graph learning concerns uncertainty about system structure.

Examples:

- edge confidence;
- observation assimilation;
- graph refinement;
- competing structural hypotheses;
- Probe history.

Future controller learning concerns expected utility of actions under context.

Examples:

- action preference;
- predicted effectiveness;
- expected risk;
- expected uncertainty;
- reserve allocation.

Therefore:

```text
Graph Learning != Controller Learning
```

Neither should be inferred automatically from the presence of `PredictionError` or `TransitionModifierSet`.

---

## 29. Current Release

**ROIF Engine v1.4.0**

Implemented or executable in the current architecture:

- stable mechanical and constraint substrate;
- pre-stressed cascade dynamics;
- dynamic physical history;
- recursive physical-state evolution;
- Counterfactual Engine;
- explicit causal-role separation;
- Active Probe Engine;
- incomplete-graph adapter;
- Vector Probe evidence;
- utilization-aware evidence;
- temporal active-evidence controls;
- Active Cascade;
- predictive-control machinery;
- finite control reserve;
- prediction-error evaluation;
- structured memory representation;
- memory-to-transition derivation;
- `TransitionModifierSet`;
- executable `PRESTRESS_TRANSFER` integration;
- matched-current-state history controls;
- history-conditioned redistribution benchmarks;
- restricted predictive-preconfiguration benchmarks;
- Temporal Image reconstruction and trajectory benchmarks;
- reproducibility and claim-boundary tests.

Validation checkpoint:

```text
10877 passed
```

---

## 30. Current Architectural Frontier

The current frontier is not a single transition from prediction error to learning.

After `v1.4.0`, several research branches remain distinct:

```text
                         v1.4.0
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
Broader Memory-       Temporal / Predictive   Recursive Active
Conditioned Channels     Validation             Exploration
        |                   |                   |
        v                   v                   v
More Explicit        Stronger Ablations       Multi-Step Probe
Transition Targets   and Transfer Tests       Planning
        |
        v
Matched-State
Identifiability
```

Separate future tracks include:

- additional transition channels beyond `PRESTRESS_TRANSFER`;
- stronger history identifiability;
- Temporal Image generalization;
- predictive-preconfiguration falsification and transfer;
- Experience Trace;
- persistent controller memory;
- experience-dependent candidate evaluation;
- graph learning;
- recursive active exploration;
- cross-domain and robotics validation.

These are research directions, not current v1.4.0 claims.

---

## 31. Architectural Invariants

The following boundaries must remain explicit as ROIF evolves:

```text
D_origin != D_fast != D_root != Node* != Probe*

GraphUpdateProposal != Graph Mutation

Large Passive Response != Directional Causal Confirmation

First Visible Response != Final Causal Decision

Useful Intervention != Structural Root

PredictedOutcome != ObservedOutcome

Prediction Error != Learning

Physical Memory != Structured Retained Memory

Structured Retained Memory != Controller Learning

Graph Learning != Controller Learning

Observed Stability != Adequate Control Reserve

TransitionModifierSet != General History-Conditioned SystemImage Operator

Temporal Image Reconstruction != General Future Forecasting

Restricted Predictive Preconfiguration != Universal Predictive Stabilization
```

Roles or mechanisms may coincide in a particular experiment, but the architecture must not assume their equivalence.

---

## 32. Scientific Claim Boundary

ROIF Engine `v1.4.0` provides executable and tested mechanisms for:

- pre-stressed cascade dynamics;
- active evidence acquisition;
- directional and temporal evidence evaluation;
- explicit causal-role separation;
- counterfactual intervention comparison;
- predictive candidate evaluation;
- finite control reserve;
- recursive physical-state evolution;
- state-mediated history dependence;
- structured memory-to-transition derivation;
- bounded transition conditioning through the tested `PRESTRESS_TRANSFER` channel;
- matched-state history controls;
- history-conditioned redistribution experiments;
- restricted anticipatory prestress preconfiguration;
- Temporal Image reconstruction and trajectory experiments.

ROIF Engine `v1.4.0` does **not** establish:

- a general history-conditioned operator over the complete `SystemImage`;
- complete future Temporal Image prediction;
- general scalar forecasting superiority;
- objective-independent whole-system predictive stabilization;
- persistent adaptive controller learning;
- biological learning;
- consciousness;
- human cognition;
- clinical validity from computational benchmarks alone;
- universal stability.

---

## 33. Guiding Principle

ROIF separates structure, mechanics, material behavior, physical history, structured retained memory, active exploration, causal evidence, graph adaptation, causal inference, system evolution, prediction, intervention, and future learning.

Complex behavior should emerge from explicit interactions among modular components rather than from hidden equivalences or monolithic logic.

The architectural objective is not to maximize the number of claimed capabilities.

It is to make each mechanism explicit enough that it can be independently implemented, tested, falsified, reproduced, and bounded.
