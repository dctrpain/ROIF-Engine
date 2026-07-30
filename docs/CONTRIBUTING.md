# Contributing to ROIF Engine

## Overview

ROIF Engine is an experimental biomechanical simulation framework.

Contributions are welcome when they improve:

- correctness;
- clarity;
- test coverage;
- numerical stability;
- documentation;
- reproducibility;
- extensibility.

All changes should preserve the modular architecture of the engine.

---

# Development Principles

Contributors should follow several core principles.

## Minimal Changes

Prefer the smallest change that solves the identified problem.

Avoid unrelated refactoring within bug-fix pull requests.

## Separation of Responsibilities

Keep the existing separation between:

- geometry;
- material behavior;
- numerical integration;
- constraints;
- network management.

Mechanical behavior belongs in material classes.

Geometric relationships belong in elements or constraints.

Simulation orchestration belongs in the network and solver layers.

## Tests Before Refactoring

Before changing existing behavior:

1. identify the current expected behavior;
2. add or confirm a test;
3. make the smallest required change;
4. run the complete test suite.

## Scientific Transparency

New physical behavior must clearly document:

- assumptions;
- equations;
- parameters;
- units;
- limitations;
- validation status.

A numerical result should not be presented as scientifically validated unless supporting validation exists.

---

# Getting Started

Clone the repository:

```bash
git clone https://github.com/dctrpain/ROIF-Engine.git
cd ROIF-Engine
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest
```

---

# Repository Structure

```text
roif_engine/
├── core/
│   ├── node.py
│   ├── element.py
│   ├── network.py
│   ├── solver.py
│   ├── constraint.py
│   ├── distance_constraint.py
│   ├── material.py
│   ├── muscle_material.py
│   ├── ligament_material.py
│   ├── fascia_material.py
│   ├── tendon_material.py
│   └── remodeling.py
│
├── tests/
│
├── docs/
│
└── examples/
```

Before modifying a component, review its corresponding documentation.

Relevant documents include:

- `ARCHITECTURE.md`
- `MATERIALS.md`
- `SOLVER.md`
- `XPBD.md`
- `TESTING.md`
- `API.md`
- `THEORY.md`
- `VALIDATION.md`

---

# Types of Contributions

## Bug Fixes

A bug fix should include:

- a clear description of the defect;
- a test that reproduces the defect when practical;
- the smallest reasonable correction;
- confirmation that existing tests still pass.

Do not combine a bug fix with broad architectural changes unless they are inseparable.

## New Material Models

A new material model should:

- inherit from the common material interface;
- preserve the material lifecycle;
- define its physical assumptions;
- document parameter units;
- include isolated tests;
- include limiting-case tests;
- identify whether it has been numerically or experimentally validated.

The required lifecycle is:

```text
begin_step()
    ↓
force()
    ↓
end_step()
```

Material state should be updated only at the appropriate lifecycle stage.

## New Constraints

A new constraint should:

- follow the existing constraint interface;
- define the geometric condition being enforced;
- document compliance behavior;
- handle fixed nodes correctly;
- include convergence and stability tests;
- describe known numerical limitations.

## Solver Changes

Solver changes require particular care.

Any modification to integration or XPBD behavior should include tests for:

- stability;
- convergence;
- deterministic execution;
- constraint error;
- energy behavior where applicable;
- sensitivity to time step size.

Solver changes should not silently alter material semantics.

## Documentation

Documentation changes should:

- describe implemented behavior;
- avoid unsupported claims;
- use consistent terminology;
- preserve one responsibility per document;
- include examples where they improve clarity.

Documentation should not claim that a planned feature is already implemented.

---

# Testing Requirements

All contributions must pass the existing test suite.

Run:

```bash
pytest
```

A new feature should normally include one or more new tests.

Tests should be:

- deterministic;
- isolated;
- readable;
- physically interpretable;
- independent of execution order.

Use tolerances for floating-point comparisons.

Avoid assertions that depend on exact binary equality unless exact equality is intentional.

---

# Expected Failures

A test may be marked as an expected failure only when it represents:

- explicitly planned behavior;
- a known and documented missing feature;
- a known limitation that should remain visible.

Expected failures must include a clear reason.

They must not be used to hide regressions.

Example:

```python
@pytest.mark.xfail(reason="True creep is not implemented yet.")
def test_true_creep():
    ...
```

When the feature is implemented, the expected-failure marker should be removed.

---

# Backward Compatibility

The public API is experimental before version `1.0.0`.

Breaking changes may occur, but they should be:

- intentional;
- documented;
- tested;
- recorded in `CHANGELOG.md`.

Avoid unnecessary renaming of public classes, methods, and parameters.

---

# Coding Style

Contributors should follow the project coding standard described in:

```text
docs/CODING_STANDARD.md
```

Until that document is finalized, use the following minimum rules:

- use clear Python names;
- prefer explicit code over clever code;
- add type hints where practical;
- keep functions focused;
- avoid hidden global state;
- document public classes and methods;
- preserve deterministic behavior.

---

# Commit Guidelines

Commits should be focused and understandable.

Recommended commit message format:

```text
type: concise description
```

Examples:

```text
fix: correct passive force dispatch order
test: add fatigue recovery coverage
docs: document XPBD solver behavior
feat: add nonlinear ligament material
refactor: simplify network force accumulation
```

Recommended commit types:

- `fix`
- `feat`
- `test`
- `docs`
- `refactor`
- `chore`

Avoid vague messages such as:

```text
update
changes
fix stuff
```

---

# Pull Requests

A pull request should explain:

1. what changed;
2. why the change was necessary;
3. which tests were added or updated;
4. whether the public API changed;
5. whether documentation was updated;
6. what limitations remain.

Keep pull requests focused on one logical change whenever possible.

---

# Review Checklist

Before submitting a contribution, confirm:

- [ ] The change has a clear purpose.
- [ ] Existing architecture boundaries are preserved.
- [ ] The full test suite passes.
- [ ] New behavior has test coverage.
- [ ] Floating-point comparisons use appropriate tolerances.
- [ ] Public API changes are documented.
- [ ] Scientific assumptions are stated explicitly.
- [ ] Unsupported clinical claims are not introduced.
- [ ] `CHANGELOG.md` is updated when appropriate.
- [ ] No unrelated files are included.

---

# Scientific and Clinical Scope

ROIF Engine is a computational biomechanics project.

Contributions must distinguish between:

- mechanical simulation;
- biological interpretation;
- clinical interpretation.

The engine may be used to test hypotheses, but simulation output alone does not establish clinical causality.

Clinical recommendations do not belong in the engine core.

---

# Security and Sensitive Data

Do not commit:

- patient information;
- personal medical data;
- credentials;
- API keys;
- private access tokens;
- confidential datasets;
- proprietary material without permission.

Use synthetic or properly anonymized data in examples and tests.

---

# Licensing

By contributing to ROIF Engine, contributors agree that their work may be distributed under the license used by the repository.

The repository license should be reviewed before submitting substantial contributions.

---

# Guiding Principle

Every contribution should make ROIF Engine more correct, understandable, reproducible, or scientifically useful without introducing unnecessary complexity.