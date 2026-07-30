# Coding Standard

## Overview

This document defines the coding conventions used throughout ROIF Engine.

The primary goals are:

- readability;
- maintainability;
- reproducibility;
- deterministic behavior;
- scientific transparency.

Code should be written for future contributors as much as for the current developer.

---

# General Principles

The project follows several guiding principles.

1. Prefer clarity over cleverness.
2. Prefer explicit behavior over implicit behavior.
3. Prefer small functions over large functions.
4. Prefer composition over duplication.
5. Preserve deterministic execution whenever possible.

Readable code is considered more valuable than compact code.

---

# Python Version

The project targets modern Python 3.

Use language features that improve readability, but avoid unnecessary complexity.

---

# Naming Conventions

## Classes

Use PascalCase.

Examples:

```python
class Node:
    ...

class Material:
    ...

class DistanceConstraint:
    ...
```

---

## Functions

Use snake_case.

```python
def compute_force():
    ...

def update_material():
    ...
```

---

## Variables

Use descriptive names.

Prefer:

```python
current_length
extension
rest_length
stiffness
compliance
time_step
```

Avoid:

```python
x
tmp
a1
foo
bar
```

except for short-lived mathematical variables.

---

## Constants

Use UPPER_CASE.

```python
DEFAULT_TIME_STEP
MAX_ITERATIONS
GRAVITY
```

---

# Type Hints

Use type hints whenever practical.

Example:

```python
def compute_force(
    extension: float,
    velocity: float
) -> float:
    ...
```

Type hints improve readability and tooling support.

---

# Function Design

Each function should perform one logical task.

Prefer:

```python
compute_force()

update_state()

apply_constraint()
```

instead of one large function performing multiple unrelated operations.

---

# Class Responsibilities

Each class should have one primary responsibility.

Examples:

| Class | Responsibility |
|--------|----------------|
| Node | Store state |
| Element | Connect nodes |
| Material | Compute constitutive response |
| Solver | Advance simulation |
| Constraint | Enforce geometry |
| Network | Coordinate simulation |

Avoid mixing responsibilities.

---

# Documentation

Public classes should include docstrings.

Example:

```python
class Node:
    """
    Represents a point mass in the biomechanical network.
    """
```

Public methods should explain:

- purpose;
- parameters;
- return values;
- units where applicable.

---

# Units

Physical quantities should use consistent SI units whenever possible.

Examples:

| Quantity | Unit |
|----------|------|
| Length | meter |
| Mass | kilogram |
| Time | second |
| Force | newton |
| Stiffness | N/m |

If another unit system is used, it should be clearly documented.

---

# Floating-Point Arithmetic

Avoid equality comparisons.

Prefer:

```python
abs(a - b) < tolerance
```

instead of:

```python
a == b
```

Use appropriate tolerances in numerical code and tests.

---

# Error Handling

Raise informative exceptions.

Prefer:

```python
raise ValueError("Rest length must be positive.")
```

instead of:

```python
raise Exception()
```

Exception messages should help identify the problem.

---

# Immutability

Do not modify data unexpectedly.

Functions should avoid hidden side effects unless mutation is their explicit purpose.

---

# Material Lifecycle

All material implementations must preserve the common lifecycle.

```text
begin_step()
    ↓
force()
    ↓
end_step()
```

Do not bypass lifecycle methods.

---

# Numerical Stability

Avoid algorithms that:

- accumulate unnecessary numerical error;
- depend on execution order;
- produce nondeterministic behavior.

Prefer stable formulations whenever practical.

---

# Performance

Correctness comes before optimization.

Optimize only after identifying actual bottlenecks.

Avoid premature optimization.

---

# Testing

Every bug fix should include a corresponding regression test whenever feasible.

New functionality should include:

- unit tests;
- edge-case tests;
- limiting-case tests.

---

# Imports

Group imports in the following order:

```python
import math
import typing

import numpy as np

from roif_engine.core.node import Node
```

Avoid wildcard imports.

Do not use:

```python
from module import *
```

---

# Magic Numbers

Avoid unexplained numeric literals.

Prefer:

```python
DEFAULT_ITERATIONS = 20
```

instead of:

```python
iterations = 20
```

Document constants when they have physical meaning.

---

# Comments

Write comments that explain **why**, not **what**.

Good:

```python
# XPBD requires compliance to remain independent of timestep.
```

Poor:

```python
# Increment i.
i += 1
```

The code should already explain what it does.

---

# Formatting

Use consistent formatting.

Recommended:

- 4 spaces indentation;
- one statement per line;
- blank lines between logical sections;
- readable line lengths.

Automatic formatting tools may be adopted in future releases.

---

# Logging

Error messages should be informative.

Debug output should not remain in production code unless intentionally configurable.

Avoid excessive console printing.

---

# Backward Compatibility

Public interfaces should remain stable whenever practical.

If a breaking change is required:

- document it;
- update tests;
- update documentation;
- record it in `CHANGELOG.md`.

---

# Scientific Transparency

Every implemented model should make its assumptions explicit.

Document:

- governing equations;
- parameter meanings;
- numerical limitations;
- validation status.

Avoid undocumented empirical constants.

---

# Code Review Checklist

Before merging code, verify:

- [ ] Names are descriptive.
- [ ] Responsibilities are well separated.
- [ ] Functions remain small.
- [ ] Public API is documented.
- [ ] Type hints are appropriate.
- [ ] Tests pass.
- [ ] Numerical assumptions are documented.
- [ ] No unnecessary complexity was introduced.
- [ ] No dead code remains.
- [ ] Floating-point comparisons use tolerances.

---

# Guiding Principle

Every line of code should make the engine easier to understand, verify, and extend.