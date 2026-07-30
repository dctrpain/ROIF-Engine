# XPBD Constraint System

## Overview

ROIF Engine uses Extended Position-Based Dynamics (XPBD) to enforce geometric constraints while maintaining numerical stability.

XPBD extends traditional Position-Based Dynamics by introducing compliance, allowing constraints to represent both rigid and deformable behavior in a physically meaningful way.

The constraint system is independent of constitutive material models.

---

# Purpose

Constraints are used to preserve geometric relationships within the biomechanical network.

Typical applications include:

- fixed distances;
- attachment points;
- joint limits;
- structural stabilization.

Constraints describe geometry, not material behavior.

---

# Design Philosophy

ROIF Engine separates:

- geometry;
- constitutive mechanics;
- numerical integration.

This separation keeps the simulation modular and extensible.

---

# Constraint Lifecycle

Each simulation step follows the sequence:

```
Predict Positions

↓

Evaluate Constraints

↓

Compute Constraint Corrections

↓

Apply Corrections

↓

Continue Simulation
```

Constraint solving occurs independently of material force computation.

---

# Compliance

Unlike classical PBD, XPBD introduces compliance.

Compliance allows constraints to behave as:

- perfectly rigid;
- nearly rigid;
- compliant;
- highly flexible.

The stiffness of a material is therefore not encoded directly into the constraint itself.

---

# Current Constraint Types

Implemented:

| Constraint | Status |
|------------|:------:|
| Distance Constraint | ✅ |

Planned:

| Constraint | Status |
|------------|:------:|
| Fixed Point | ⏳ |
| Hinge | ⏳ |
| Angular | ⏳ |
| Volume | ⏳ |
| Surface | ⏳ |

---

# Distance Constraint

The current implementation preserves the distance between two connected nodes.

This forms the basis for element stabilization and biomechanical network integrity.

---

# Advantages of XPBD

Compared with classical PBD:

- improved stability;
- compliance support;
- better convergence;
- physically interpretable stiffness;
- improved behavior under large deformations.

---

# Future Extensions

Future development may include:

- articulated joints;
- soft-body constraints;
- contact constraints;
- collision handling;
- volume preservation;
- anisotropic constraints.

---

# Design Principle

XPBD is responsible only for maintaining geometric relationships.

Mechanical forces remain the responsibility of the material subsystem.

This separation preserves modularity and simplifies future expansion of the engine.