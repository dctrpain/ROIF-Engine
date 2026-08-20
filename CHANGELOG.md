# Changelog

All notable changes to the ROIF Engine project will be documented in this file.

The format follows the principles of *Keep a Changelog*.

---

# [1.4.0] - 2026-08-20

## Added

### Structured Memory-to-Transition Layer

* Added `roif/history/transition_modifiers.py`.
* Added explicit bounded transition modifiers with structured provenance.
* Added transition-modifier channels and deterministic modifier signatures.
* Added `TransitionModifierSet` as a policy-free transition-conditioning interface.
* Added explicit separation between structured memory provenance and direct physical-state evolution.

### Memory Transition Derivation

* Added `roif/history/memory_transition_derivation.py`.
* Added deterministic derivation of bounded transition modifiers from structured memory evidence.
* Added explicit semantic-to-physical target bindings.
* Added binding polarity and target-channel mapping.
* Added provenance-preserving derivation signatures.
* Added rejection of ambiguous duplicate physical transition targets.

### System Evolution Integration

* Integrated `TransitionModifierSet` into `roif/history/system_evolution.py`.
* Added executable support for the `PRESTRESS_TRANSFER` transition channel.
* Added bounded modification of effective prestress-transfer gain.
* Preserved the exact legacy transition path when modifiers are absent or zero.
* Added rejection of unsupported active modifier channels.
* Added transition-modifier provenance to evolution metadata.
* Kept the integration explicitly narrower than a general history-conditioned operator over the complete `SystemImage`.

### Matched-Current-State Identifiability

* Added `roif_matched_state_history_operator_identifiability_benchmark.py`.
* Added a matched-current-state physical-history control.
* Added explicit separation between:
  * history embodied in current physical organization;
  * structured memory retained outside the matched physical state;
  * independently identifiable transition conditioning.
* Added null-result handling when no independent physical-history contribution remains after represented current state is matched.

### Structured-Memory-Conditioned Transition Benchmark

* Added `roif_memory_conditioned_matched_state_transition_benchmark.py`.
* Added matched-current-state experiments in which represented physical state, topology, probe event, attractor, and binding are held fixed while structured conditioning memory varies.
* Demonstrated different bounded memory-derived transition modifiers under matched represented current state.
* Demonstrated different subsequent prestress transitions through the implemented `PRESTRESS_TRANSFER` channel.
* Preserved matched adaptive connection state across compared conditions.

### History-Conditioned Redistribution Controls

* Added Q8 history-conditioned redistribution benchmarks.
* Added matched-state decomposition of prestress-state and adaptive-connection contributions.
* Added capacity-aware redistribution analysis.
* Added explicit distinction between redistribution and stabilization.

### Predictive Preconfiguration Audits

* Added `roif_predictive_stabilization_benchmark.py`.
* Added `roif_q7_objective_independence_audit.py`.
* Added `roif_q7_off_nominal_transferability_audit.py`.
* Added restricted one-step prestress-layer predictive preconfiguration.
* Added objective-independence auditing.
* Added off-nominal transferability auditing.
* Added explicit claim boundaries preventing interpretation as general objective-independent whole-system predictive stabilization.

### Temporal Image and Trajectory Benchmarks

* Added `roif_temporal_image_reconstruction_benchmark.py`.
* Added `roif_multilayer_temporal_image_trajectory_benchmark.py`.
* Added deterministic temporal-image reconstruction controls.
* Added endpoint-versus-trajectory comparisons.
* Added explicit distinction between scalar trajectory summaries and a complete future `TemporalImage`.
* Added claim-boundary tests preventing reinterpretation of trajectory analysis as full future Temporal-Image prediction.

### Reproducibility Assets

* Added benchmark result artifacts for matched-state identifiability, structured-memory transition conditioning, Q7 audits, Q8 redistribution decomposition, and temporal-image reconstruction.
* Added Entropy figure-generation support in `experiments/build_entropy_figures.py`.
* Added architecture claim-boundary regression tests.
* Added transition-modifier integration tests.
* Added memory-transition derivation tests.

## Changed

* Extended `SystemEvolution` with an optional transition-conditioning pathway while preserving backward-compatible zero-modifier behavior.
* Refined predictive-preconfiguration semantics to a restricted implemented prestress-layer mechanism.
* Refined history dependence into experimentally separable state-mediated and structured-memory-conditioned mechanisms.
* Refined temporal claims so that complete future `TemporalImage` prediction remains an architectural target rather than a demonstrated capability.

## Validation

* Full project test suite passed after integration of structured-memory-conditioned transition support and associated benchmarks.

---
# [1.3.0] - 2026-08-09

## Added

### Vector Probe Layer

* Added `roif/vector_probe.py`.
* Introduced directional active-evidence evaluation.
* Added explicit vector-alignment analysis.
* Added evidence decisions:

  * `SUPPORTS`;
  * `CONTRADICTS`;
  * `INSUFFICIENT`;
  * `NO_RESPONSE`.
* Separated scalar response amplitude from directional causal evidence.
* Added explicit rejection of reverse-direction relations.

### Utilization Layer

* Added `roif/utilization.py`.
* Introduced utilization-aware active evidence.
* Added utilization-change evaluation for Probe responses.
* Integrated utilization information into active causal assessment.

### Active Cascade

* Added `roif/active_cascade.py`.
* Added `roif/active_cascade_ape.py`.
* Introduced recursive transition through actively confirmed relations.
* Added confirmed-edge tracking.
* Added rejected-relation tracking.
* Added current-node transition state.
* Added evidence provenance and audit handling.
* Integrated Active Probe Engine decisions with active cascade progression.

### Temporal Active Evidence

* Added observation-window-aware Probe evaluation.
* Added distinction between early and full observation windows.
* Added delayed-response visibility handling.
* Prevented premature confirmation from the first visible response.
* Added temporal gating for relation authorization.

### Predictive Control

* Added `roif/predictive_control.py`.
* Introduced domain-independent predictive stabilization.
* Added:

  * `PredictiveState`;
  * `DisturbanceEstimate`;
  * `ControlReserve`;
  * `StabilizationDemand`;
  * `ControlCandidate`;
  * `PredictedOutcome`;
  * `CandidateEvaluation`;
  * `PredictiveControlDecision`;
  * `PredictionError`.
* Added predictive candidate ranking.
* Added finite control-reserve representation.
* Added stabilization-demand estimation.
* Added prediction-versus-observation comparison.
* Added support for predictive decisions:

  * `ACTION`;
  * `PROBE`;
  * `HOLD`;
  * `NO_SAFE_ACTION`.
* Added uncertainty-driven Probe selection.
* Added reversible Probe fallback when corrective action is insufficiently justified.

---

## Added вЂ” Mechanical Validation

### Pre-Stressed Mechanical Benchmarks

Added validation cases for:

* pre-stressed spring;
* serial weak link;
* pre-stressed branching;
* symmetric branching ambiguity.

### 03C вЂ” Misleading High-Amplitude Branch

Added an adversarial benchmark in which:

```text
largest scalar response
        !=
best directional evidence
```

Validated that:

* the louder branch does not win from amplitude alone;
* orthogonal response is insufficient causal evidence;
* the quieter aligned branch can be selected;
* only authorized directional evidence advances the active cascade.

### 03D вЂ” Misleading Reverse-Direction Response

Added an adversarial benchmark in which the strongest response points opposite to the proposed causal relation.

Validated that:

```text
strong reverse response
        ->
CONTRADICTS
        ->
relation rejected
```

The reverse branch is explicitly rejected rather than merely ignored.

### 03E вЂ” Delayed Misleading Response

Added a temporal adversarial benchmark.

Validated that:

* an early response may be visible without being confirmable;
* a delayed response may become visible only in the full observation window;
* early evidence does not force premature transition;
* final authorization waits for the relevant temporal evidence.

Established:

```text
First Response != Correct Response
```

---

## Added вЂ” Clinical Validation

### FootвЂ“Knee Active Probe

* Added `validation/clinical/foot_knee_probe_case.py`.
* Added Active Probe validation for the FootвЂ“Knee case.
* Added recursive Probe validation.
* Added corresponding integration and regression tests.

---

## Added вЂ” Sailing Validation

### 04A вЂ” Coupled YachtвЂ“Crew System

Added:

`validation/sailing/sailing_yacht_crew_coupled_case.py`

Introduced the first mixed physical-controller benchmark.

The benchmark separates:

### Physical Plant

* wind-shift disturbance;
* sail load;
* heel/yaw response;
* course error;
* rudder action;
* sail-trim action.

### Living Controller

* helmsman control demand;
* sail-trimmer control demand.

The benchmark uses published yachtвЂ“crew modeling literature as the basis for the coupled-system topology while keeping ROIF-specific normalized parameters separate from published claims.

### 04A вЂ” YachtвЂ“Crew Predictive Control

Added:

`validation/sailing/sailing_yacht_crew_predictive_control.py`

Integrated the sailing benchmark with the domain-neutral Predictive Control layer.

Added candidate policies for:

* helm correction;
* sail-trim correction;
* coupled helm + trim correction;
* small information Probe;
* hold.

The benchmark evaluates candidate actions before execution using predicted residual state, uncertainty, cost, safety, and available control reserve.

Under the current normalized benchmark state, the coupled helmвЂ“trim action is selected.

---

## Changed

### Active Probe Semantics

Extended Active Probe from simple information acquisition toward directional and temporal causal evidence.

Probe response magnitude alone is no longer sufficient for relation confirmation.

Established:

```text
Passive Amplitude != Active Causal Evidence
```

and:

```text
Large Response != Correct Direction
```

### Active Transition Semantics

Causal evidence can now produce different structural consequences:

```text
SUPPORTS
    ->
candidate may be confirmed

CONTRADICTS
    ->
candidate may be rejected

INSUFFICIENT
    ->
no confirmation

NO_RESPONSE
    ->
no confirmation
```

All state-changing transitions remain authorization-controlled.

### Root Detection

* Updated `roif_root_detector.py`.
* Continued separation of structural mediation from intervention utility.
* Preserved independent causal roles.

### Predictive Stabilization Semantics

Introduced explicit separation between:

```text
Observed Stability
```

and:

```text
Available Stabilization Reserve
```

A regulated output may remain controlled while compensatory reserve is under pressure.

---

## Architectural Milestones

ROIF now separates the following concepts:

```text
D_origin != D_fast != D_root != Node*
```

```text
D_root != Node* != Probe*
```

```text
Structural Mediation != Intervention Utility
```

```text
GraphUpdateProposal != Graph Mutation
```

```text
Passive Amplitude != Active Causal Evidence
```

```text
High Amplitude != Directional Support
```

```text
First Response != Correct Response
```

```text
PredictedOutcome != ObservedOutcome
```

```text
Observed Stability != Adequate Control Reserve
```

---

## Validation

Added or extended tests for:

* Active Cascade;
* Active Cascade + APE integration;
* utilization;
* Vector Probe;
* pre-stressed spring ground truth;
* serial weak-link ground truth;
* serial weak-link Active Cascade;
* branching ground truth;
* branching Active Probe;
* symmetric branching ambiguity;
* misleading high-amplitude response;
* reverse-direction contradiction;
* delayed-response temporal control;
* FootвЂ“Knee Active Probe;
* FootвЂ“Knee recursive Probe;
* Predictive Control;
* sailing yachtвЂ“crew Predictive Control.

Recent targeted validation checkpoints include:

```text
Mechanical / Active Probe regression:
505 passed
```

```text
Predictive Control:
63 passed
```

```text
04A YachtвЂ“Crew Predictive Control:
76 passed
```

---

## Repository Checkpoints

Important development commits:

```text
fc08f1b
validation: add active probe cascade controls and 04A yacht-crew benchmark
```

```text
c9d2cfe
validation: add predictive control layer and 04A yacht-crew predictive benchmark
```

---

## Scientific Milestone

ROIF evolved beyond active exploration of incomplete causal graphs.

Version 1.3.0 introduces three additional capabilities:

```text
ACTIVE EVIDENCE
      в†“
direction and timing determine whether
a response supports or contradicts a relation

ACTIVE CASCADE
      в†“
authorized evidence can advance or reject
candidate causal transitions

PREDICTIVE CONTROL
      в†“
the system can evaluate possible actions
before committing to them
```

The 04A yachtвЂ“crew benchmark also introduces the first explicit ROIF validation of a coupled system containing both a physical plant and an adaptive living controller.

The current architectural frontier is now:

```text
Prediction
    в†“
Action / Probe
    в†“
Observation
    в†“
Prediction Error
    в†“
Experience Trace
        в†‘
       NEXT
```

Persistent controller memory and experience-dependent adaptation are not yet implemented.

---

# [1.2.0] - 2026-08-06

## Added

### Active Probe Engine

* Introduced the Active Probe Engine (APE) architecture.
* Added `probe_entities.py`.
* Added `probe_registry.py`.
* Added `probe_policy.py`.
* Added `probe_planner.py`.
* Added `probe_graph_adapter.py`.
* Added `active_probe_engine.py`.

### Validation

* Added the first clinical validation package.
* Added FootвЂ“Knee clinical validation case.
* Added Active Probe integration tests.
* Added Active Probe end-to-end tests.
* Added role validation tests for D_origin, D_fast, D_root and Node*.

## Changed

* Redesigned D_root using structural mediation instead of intervention utility.
* Fully separated D_origin, D_fast, D_root and Node*.
* Improved counterfactual collateral-effect evaluation.
* Refactored root detection workflow.
* Extended validation pipeline.

## Validation

* Clinical validation pipeline completed.
* Active Probe Engine fully integrated.
* End-to-End validation completed.
* Complete regression suite passing.

## Statistics

* 19 new source files.
* More than 15,000 new lines of implementation and tests.
* More than 6,000 automated tests passing.

## Scientific Milestone

ROIF evolved from a passive cascade analysis engine into a framework capable of actively exploring incomplete causal graphs before causal inference.

---

# [1.1.0] - 2026

## Added

### Dynamic System Layer

* History Engine.
* Rheological Memory.
* Counterfactual Engine.
* Recursive intervention analysis.
* End-to-End execution pipeline.

### Solver

* Stable recursive cascade solver.
* Counterfactual scenario evaluation.
* Recursive intervention comparison.

### Validation

* Full integration tests.
* End-to-End testing.

## Changed

* Solver stabilization.
* Improved recursive propagation.
* Improved tensor propagation.
* Improved simulation reproducibility.

## Scientific Milestone

ROIF evolved from a static cascade simulator into a dynamic framework capable of evaluating alternative intervention scenarios.

---

# [1.0.0] - 2026

## Added

### Core Engine

* Stable graph architecture.
* Node framework.
* Element framework.
* Material framework.
* Capacity Tensor.
* Constraint system.
* Recursive Cascade Solver.

### Materials

* Elastic materials.
* Muscle material.
* Ligament material.
* Fascia material.
* Tendon material.
* Remodeling engine.

### Visualization

* Viewer.
* Network visualization.
* Snapshot generation.

### Validation

* Large automated regression suite.
* Structural simulation tests.
* Material behaviour tests.
* Visualization tests.

## Scientific Milestone

ROIF became a deterministic engine for cascade propagation in pre-stressed systems.

---

# [0.1.0-alpha] - 2026-07-29

## Added

### Initial Architecture

* Initial project architecture.
* Node class.
* Element class.
* Material abstraction.
* Network management.
* Solver.
* XPBD constraint framework.

### Material Models

* Elastic response.
* Viscoelastic damping.
* Pretension.
* Fatigue.
* Recovery.
* Remodeling.
* Material failure.

### Testing

Implemented automated regression tests.

Current status at the time of release:

* 13 PASSED
* 1 XFAILED (true creep not yet implemented)

### Documentation

Added:

* README.md
* PROJECT_STATUS.md
* requirements.txt
* .gitignore
