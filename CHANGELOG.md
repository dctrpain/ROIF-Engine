# Changelog

All notable changes to the ROIF Engine project will be documented in this file.

The format follows the principles of *Keep a Changelog*.

---

# [1.2.0] - 2026-08-06

## Added

### Active Probe Engine

- Introduced the Active Probe Engine (APE) architecture.
- Added `probe_entities.py`.
- Added `probe_registry.py`.
- Added `probe_policy.py`.
- Added `probe_planner.py`.
- Added `probe_graph_adapter.py`.
- Added `active_probe_engine.py`.

### Validation

- Added the first clinical validation package.
- Added Foot–Knee clinical validation case.
- Added Active Probe integration tests.
- Added Active Probe end-to-end tests.
- Added role validation tests for D_origin, D_fast, D_root and Node*.

## Changed

- Redesigned D_root using structural mediation instead of intervention utility.
- Fully separated D_origin, D_fast, D_root and Node*.
- Improved counterfactual collateral-effect evaluation.
- Refactored root detection workflow.
- Extended validation pipeline.

## Validation

- Clinical validation pipeline completed.
- Active Probe Engine fully integrated.
- End-to-End validation completed.
- Complete regression suite passing.

## Statistics

- 19 new source files.
- More than 15,000 new lines of implementation and tests.
- More than 6,000 automated tests passing.

## Scientific Milestone

ROIF evolved from a passive cascade analysis engine into a framework capable of actively exploring incomplete causal graphs before causal inference.

---

# [1.1.0] - 2026

## Added

### Dynamic System Layer

- History Engine.
- Rheological Memory.
- Counterfactual Engine.
- Recursive intervention analysis.
- End-to-End execution pipeline.

### Solver

- Stable recursive cascade solver.
- Counterfactual scenario evaluation.
- Recursive intervention comparison.

### Validation

- Full integration tests.
- End-to-End testing.

## Changed

- Solver stabilization.
- Improved recursive propagation.
- Improved tensor propagation.
- Improved simulation reproducibility.

## Scientific Milestone

ROIF evolved from a static cascade simulator into a dynamic framework capable of evaluating alternative intervention scenarios.

---

# [1.0.0] - 2026

## Added

### Core Engine

- Stable graph architecture.
- Node framework.
- Element framework.
- Material framework.
- Capacity Tensor.
- Constraint system.
- Recursive Cascade Solver.

### Materials

- Elastic materials.
- Muscle material.
- Ligament material.
- Fascia material.
- Tendon material.
- Remodeling engine.

### Visualization

- Viewer.
- Network visualization.
- Snapshot generation.

### Validation

- Large automated regression suite.
- Structural simulation tests.
- Material behaviour tests.
- Visualization tests.

## Scientific Milestone

ROIF became a deterministic engine for cascade propagation in pre-stressed systems.

---

# [0.1.0-alpha] - 2026-07-29

## Added

### Initial Architecture

- Initial project architecture.
- Node class.
- Element class.
- Material abstraction.
- Network management.
- Solver.
- XPBD constraint framework.

### Material Models

- Elastic response.
- Viscoelastic damping.
- Pretension.
- Fatigue.
- Recovery.
- Remodeling.
- Material failure.

### Testing

Implemented automated regression tests.

Current status at the time of release:

- 13 PASSED
- 1 XFAILED (true creep not yet implemented)

### Documentation

Added:

- README.md
- PROJECT_STATUS.md
- requirements.txt
- .gitignore