# ROIF Engine

> **Recursive Organic Integration Framework**
>
> Universal Cascade Analysis, Active Exploration, History-Conditioned Transition, and Predictive Control Engine

**Current release:** `v1.4.0`
**Release date:** 2026-08-20
**Release commit:** `580716c13fd4866484cbb651157002123e337b06`
**Regression suite:** `10877 passed`

---

## Overview

ROIF Engine is a domain-independent computational architecture for representing, probing, evolving, and experimentally separating different mechanisms of response in **pre-stressed complex systems**.

Unlike conventional graph-analysis pipelines, ROIF is designed to work with incomplete causal structure, dynamically loaded systems, recursive cascade propagation, active evidence, physical history, structured memory, and explicitly bounded predictive-control mechanisms.

The current architecture can:

* propagate disturbances through a represented pre-stressed system;
* distinguish structural origin, early functional failure, cascade mediation, and intervention roles;
* actively probe incomplete causal structure;
* evaluate directional, utilization, and temporal evidence;
* update an active cascade only through authorized evidence;
* evolve physical state recursively;
* preserve and test state-mediated path dependence;
* represent structured memory outside the matched current physical state;
* derive bounded transition modifiers from structured memory;
* apply the currently implemented modifier path through `PRESTRESS_TRANSFER`;
* compare matched-current-state systems with different retained histories;
* evaluate restricted predictive preconfiguration;
* reconstruct and compare Temporal Image trajectories under controlled benchmark conditions;
* preserve explicit scientific claim boundaries around each implemented mechanism.

Medicine and biomechanics remain important validation domains, but they are not architectural boundaries.

---

# Why ROIF?

Many causal-analysis systems effectively ask:

> **"Given the graph, what is the cause?"**

ROIF separates a broader sequence of questions:

> **"How can a disturbance propagate through the currently represented system?"**

> **"What is the most informative next observation if causal structure is incomplete?"**

> **"Does the observed response actually support the proposed direction of propagation?"**

> **"Can two systems with the same current measured state respond differently because their retained histories differ?"**

> **"What bounded action or preconfiguration is justified under the current load, reserve, uncertainty, and tested objective?"**

These questions are deliberately kept separate. A large response is not automatically causal evidence; a useful intervention is not automatically the structural root; history dependence is not automatically a general forecasting capability; and predictive preconfiguration is not equivalent to universal predictive stabilization.

---

# Core Causal Roles

ROIF distinguishes four causal and control roles.

| Role | Meaning |
| --- | --- |
| **D_origin** | Structural origin of the cascade |
| **D_fast** | Earliest observable functional failure |
| **D_root** | Structural mediation node maintaining cascade propagation |
| **Node*** | Best counterfactual intervention point |

These roles are intentionally independent. A node may occupy more than one role in a particular system, but the architecture does not assume that the roles coincide.

---

# Cascade Engine

The Cascade Engine performs recursive propagation through represented pre-stressed systems.

Current capabilities include:

* recursive cascade propagation;
* Capacity Tensor analysis;
* vector-aware transport;
* utilization analysis;
* counterfactual simulation;
* mediation-based root detection;
* independent intervention ranking;
* structural and physical history mechanisms.

The passive cascade layer describes how disturbances can propagate through the currently represented system. It does not by itself establish that an observed active response confirms the causal direction under investigation.

---

# Active Probe Engine

The Active Probe Engine allows ROIF to investigate incomplete causal structure instead of assuming that the graph is already known.

```text
Incomplete Graph
      |
      v
Active Probe Engine
      |
      v
Probe*
      |
      v
Observation
      |
      v
Evidence
      |
      v
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

ROIF separates passive transport from the response produced by an active probe.

A branch may carry a large passive scalar response while producing an active response that is aligned, orthogonal, reversed, delayed, insufficient, or absent relative to the investigated relation.

```text
Passive Cascade
      |
      v
Candidate Relations
      |
      v
Active Probe
      |
      v
Observed Response Vector
      |
      v
Directional / Temporal Evidence
      |
      v
Active Cascade Update
```

The Vector Probe layer evaluates active evidence independently of passive amplitude. This prevents a high-amplitude response from automatically being interpreted as causal confirmation.

---

# Predictive Control

ROIF contains a domain-independent Predictive Control Layer.

The layer evaluates current state, stabilization demand, available control reserve, uncertainty, and alternative candidate actions before selecting an admissible operation.

```text
Current State
     +
Disturbance
     +
Control Reserve
      |
      v
Stabilization Demand
      |
      v
Candidate Actions
      |
      v
Predicted Outcomes
      |
      v
Candidate Ranking
      |
      v
ACTION / PROBE / HOLD / NO SAFE ACTION
```

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

Observed stability is not treated as proof of adequate reserve. Stabilization margin remains an explicit audit quantity.

---

# Prediction Error

Predictive Control preserves an explicit boundary between prediction and observation.

```text
Predicted State
      |
      v
Action / Probe
      |
      v
Observed State
      |
      v
Prediction Error
```

Prediction error records the difference between predicted and subsequently observed state variables.

Persistent experience-dependent controller learning is **not claimed as an implemented general capability in v1.4.0**.

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

Counterfactual evaluation remains separated from causal inference. A useful intervention is not automatically the structural root of a cascade.

---

# Physical History and Recursive System Evolution

ROIF contains mechanisms in which previous loading changes the physical organization of the system itself.

Examples include:

* fatigue;
* recovery;
* remodeling;
* rheological memory;
* irreversible material adaptation;
* adaptive connection state;
* recursive physical-state evolution.

This produces **state-mediated history dependence**: prior events can alter the present physical state, and the altered present state can change subsequent response.

This mechanism does not require a separate retained memory object. History may already be embodied in the current physical organization.

---

# Structured Memory-to-Transition Layer вЂ” v1.4.0

Version `v1.4.0` adds a second experimentally separable mechanism.

ROIF can retain structured memory evidence outside the matched current physical state, derive bounded transition modifiers from that evidence, and use those modifiers to condition a subsequent transition through the currently implemented physical channel.

```text
Historical Evidence
      |
      v
Structured Memory
      |
      v
Memory-to-Transition Derivation
      |
      v
TransitionModifierSet
      |
      v
PRESTRESS_TRANSFER
      |
      v
Subsequent SystemEvolution
```

The implementation includes:

* `roif/history/memory_transition_derivation.py`;
* `roif/history/transition_modifiers.py`;
* `TransitionModifierSet`;
* explicit semantic-to-physical target bindings;
* binding polarity and target-channel mapping;
* bounded transition modifiers;
* structured provenance;
* deterministic modifier signatures;
* rejection of ambiguous duplicate physical transition targets;
* integration with `roif/history/system_evolution.py`;
* executable support for the `PRESTRESS_TRANSFER` channel;
* exact preservation of the legacy transition path when modifiers are absent or zero;
* explicit rejection of unsupported active modifier channels.

## Critical boundary

The current implementation is intentionally narrower than a general history-conditioned operator over the complete `SystemImage`.

```text
TransitionModifierSet
        !=
general history-conditioned SystemImage operator
```

The demonstrated result is therefore:

> structured retained memory can condition a tested subsequent physical transition through an explicit bounded mechanism.

It is **not** a demonstration that arbitrary historical information can modify every system operator or predict the complete future state.

---

# Matched-Current-State Controls

A central v1.4.0 experimental strategy is to compare systems that are matched on the relevant current measured physical state while differing in retained history.

This separates two mechanisms that can otherwise be confused:

```text
History A -> Present Physical State A -> Future Response
History B -> Present Physical State B -> Future Response
```

from the stricter control:

```text
History A -> Matched Present State + Memory A
History B -> Matched Present State + Memory B
                              |
                              v
                   Different bounded modifiers
                              |
                              v
                   Subsequent transition test
```

The matched-state controls test whether retained structured history contributes information beyond the matched current physical state under the implemented transition mechanism.

---

# Predictive Preconfiguration

ROIF v1.4.0 includes restricted computational experiments in anticipatory prestress preconfiguration.

The claim is deliberately limited.

The benchmark evaluates whether a tested preconfiguration policy can improve a defined one-step objective under specified conditions and available reserve.

It does **not** establish:

* objective-independent whole-system stabilization;
* universal optimal control;
* general future-state prediction;
* complete Temporal Image prediction;
* biological anticipation;
* learned human-like predictive behavior.

Predictive preconfiguration should therefore be read as a tested architectural mechanism, not as a synonym for unrestricted forecasting.

---

# Temporal Image Benchmarks

The current repository includes Temporal Image reconstruction and multilayer temporal-trajectory benchmarks.

These experiments test separable properties of the architecture rather than a single scalar notion of "ROIF accuracy."

The temporal benchmark family includes controlled comparisons intended to distinguish effects such as:

* current-state information;
* fixed coupling;
* retained history;
* memory-free ablation;
* trajectory-conditioned reconstruction.

The experiments support analysis of trajectory dependence and reconstruction under their tested conditions.

They do **not** establish complete prediction of a future Temporal Image for arbitrary systems.

---

# History-Conditioned Redistribution

The current benchmark suite also tests history-conditioned redistribution under controlled conditions, including matched-state decomposition of contributions associated with prestress and adaptive connections.

These experiments are intended to determine which represented mechanisms carry historical information into subsequent redistribution.

They should not be interpreted as evidence for a universal history operator.

---

# Mixed-System Validation

The `04A Sailing Yacht-Crew` benchmark represents a coupled system containing a mechanical plant and a living controller.

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

The benchmark evaluates the architecture as a coupled physical-plant + living-controller system and includes alternative stabilization strategies such as helm correction, sail-trim correction, coupled action, hold, and information probe.

This is a cross-domain validation environment for the control architecture. It is not a claim that human cognition has been fully modeled.

---

# Validation

ROIF is developed using a validation-first approach.

Validation families include:

* unit tests;
* integration tests;
* end-to-end tests;
* mechanical ground-truth validation;
* clinical/biomechanical validation environments;
* Active Probe validation;
* Active Cascade validation;
* vector-response validation;
* temporal-response validation;
* counterfactual validation;
* Predictive Control validation;
* mixed yacht-crew validation;
* matched-current-state history controls;
* structured-memory-conditioned transition benchmarks;
* predictive-preconfiguration audits;
* objective-independence audits;
* off-nominal transferability audits;
* history-conditioned redistribution controls;
* Temporal Image reconstruction benchmarks;
* multilayer temporal-trajectory benchmarks;
* architecture claim-boundary tests.

For release `v1.4.0`, the complete repository regression suite completed with:

```text
10877 passed
```

---

# Reproducibility Assets

The repository contains executable benchmark scripts, stored benchmark outputs, figure-generation code, and tests supporting the current manuscript-facing computational claims.

Key experiment scripts include:

```text
experiments/
    roif_matched_state_history_operator_identifiability_benchmark.py
    roif_memory_conditioned_matched_state_transition_benchmark.py
    roif_multilayer_temporal_image_trajectory_benchmark.py
    roif_predictive_stabilization_benchmark.py
    roif_q7_objective_independence_audit.py
    roif_q7_off_nominal_transferability_audit.py
    roif_q8_history_conditioned_redistribution_benchmark.py
    roif_q8_history_conditioned_redistribution_matched_state_benchmark.py
    roif_temporal_image_reconstruction_benchmark.py
    build_entropy_figures.py
```

Stored outputs include:

```text
benchmark_results/
    matched_state_history_operator_identifiability_v1.json
    memory_conditioned_matched_state_transition_v1.json
    multilayer_temporal_image_trajectory_v1.json
    predictive_stabilization_v1.json
    q7_objective_independence_audit_v1.json
    q7_off_nominal_transferability_audit_v2.json
    q8_history_conditioned_redistribution_capacity_v1.json
    q8_history_conditioned_redistribution_matched_state_v2.json
    temporal_image_reconstruction_v1.json
    temporal_image_reconstruction_methods_v1.csv
    temporal_image_reconstruction_slices_v1.csv
```

The release also includes tests for memory-transition derivation, transition modifiers, system-evolution integration, predictive stabilization, Temporal Image reconstruction, matched-state redistribution, and architectural claim boundaries.

For manuscript reproducibility, use the immutable release tag:

```text
v1.4.0
```

corresponding to commit:

```text
580716c13fd4866484cbb651157002123e337b06
```

Repository:

https://github.com/dctrpain/ROIF-Engine

Release tag:

https://github.com/dctrpain/ROIF-Engine/tree/v1.4.0

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

    history/
        memory_transition_derivation.py
        system_evolution.py
        transition_modifiers.py

    solver.py
    network.py
    node.py
    element.py
    material.py

experiments/
benchmark_results/
validation/
tests/
docs/
```

---

# Current Architecture

```text
Pre-Stressed System
        |
        v
Passive Cascade
        |
        v
Incomplete / Candidate Structure
        |
        v
Active Probe
        |
        v
Vector + Temporal Evidence
        |
        v
Active Cascade
        |
        v
Current Dynamic State
        |
        +------------------------------+
        |                              |
        v                              v
Physical SystemEvolution       Structured Memory
        |                              |
        |                              v
        |                    Memory-to-Transition
        |                          Derivation
        |                              |
        |                              v
        |                    TransitionModifierSet
        |                              |
        +--------------+---------------+
                       |
                       v
             Subsequent Transition
                       |
                       v
              Predictive Control
                       |
             +---------+---------+
             |         |         |
             v         v         v
          Demand     Reserve   Candidates
                       |
                       v
             Action / Probe / Hold
                       |
                       v
                  Observation
                       |
                       v
                Prediction Error
```

The architecture intentionally keeps passive propagation, active causal evidence, physical history, retained structured memory, transition conditioning, prediction, observation, and control conceptually separable.

---

# Scientific Claim Boundaries

ROIF Engine `v1.4.0` provides executable and tested mechanisms for:

* recursive cascade dynamics in represented pre-stressed systems;
* active exploration of incomplete causal structure;
* vector- and temporal-sensitive active evidence;
* counterfactual intervention analysis;
* predictive-control candidate evaluation;
* recursive physical-state evolution;
* state-mediated history dependence;
* structured-memory-conditioned transition modulation through the tested `PRESTRESS_TRANSFER` channel;
* matched-current-state history controls;
* restricted anticipatory prestress preconfiguration;
* controlled Temporal Image reconstruction and trajectory experiments;
* history-conditioned redistribution experiments.

ROIF Engine `v1.4.0` does **not** establish:

* a general history-conditioned operator over the complete `SystemImage`;
* full future Temporal Image prediction;
* objective-independent whole-system predictive stabilization;
* universal stability;
* biological learning;
* autonomous psychological learning;
* consciousness;
* a complete model of human cognition;
* clinical validity as a consequence of the computational benchmarks alone.

These boundaries are part of the architecture and validation strategy, not merely documentation disclaimers.

---

# Design Principles

Every architectural decision should remain:

* domain-independent;
* mathematically formalizable;
* compatible with recursive cascade dynamics;
* compatible with pre-stressed systems;
* compatible with Active Probe;
* explicit about uncertainty;
* explicit about control reserve;
* auditable;
* deterministic where required by validation;
* scientifically defensible;
* experimentally testable;
* suitable for peer-reviewed publication.

---

# Architectural Safety

ROIF preserves explicit authorization boundaries between evidence and structural modification.

The **Non-Fonit Gate** remains a hard architectural constraint.

Information gain, predictive utility, or expected stabilization benefit must not override unacceptable cascade risk.

---

# Development Frontier

The implemented v1.4.0 memory-to-transition mechanism does not close the broader research problem of persistent adaptive controller memory.

Future work may investigate:

```text
Prediction
    |
    v
Action / Probe
    |
    v
Observed State
    |
    v
Prediction Error
    |
    v
Experience Trace
    |
    v
Persistent Controller Memory
    |
    v
Modified Predictive State
    |
    v
Next Prediction
```

Working concepts such as **Memory Scar** and **Predictive Preload** remain research terminology unless and until they are formally defined, implemented, and independently validated.

Additional development directions include broader transition channels beyond `PRESTRESS_TRANSFER`, stronger identifiability controls, external-domain replication, and further separation of architectural capability from benchmark-specific performance.

---

# License

See the `LICENSE` file for licensing information.
