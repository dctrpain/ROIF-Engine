# Solver

## Overview

The solver is responsible for advancing the biomechanical system through time.

It integrates node motion, applies internal and external forces, enforces constraints, and updates material states at each simulation step.

The solver operates independently of constitutive material behavior.

---

# Responsibilities

The solver is responsible for:

- force accumulation;
- numerical integration;
- constraint enforcement;
- material updates;
- simulation stepping.

The solver does **not** define the constitutive behavior of biological tissues.

---

# Simulation Loop

Each simulation step follows the sequence:

```
begin_step()

↓

Compute Element Forces

↓

Accumulate Forces

↓

Apply Constraints (XPBD)

↓

Integrate Motion

↓

end_step()
```

This workflow ensures a clear separation between material behavior and numerical integration.

---

# Numerical Integration

The current implementation uses explicit time integration.

Advantages:

- simple implementation;
- computational efficiency;
- deterministic behavior;
- suitable for small and medium biomechanical systems.

Limitations:

- smaller stable time steps;
- less suitable for extremely stiff systems.

---

# Force Accumulation

Each element computes its internal force through its assigned material model.

The resulting forces are accumulated at connected nodes before integration.

The solver itself does not distinguish between muscles, ligaments, fascia, or tendons.

---

# Constraint Handling

Geometric constraints are resolved using the XPBD framework.

Constraint resolution is independent of material constitutive laws.

This separation improves modularity and numerical stability.

---

# Material Updates

Before force computation:

```
begin_step()
```

After integration:

```
end_step()
```

The solver only invokes these methods.

The internal behavior of each material remains encapsulated within the material implementation.

---

# Stability

Simulation stability depends primarily on:

- time step;
- material stiffness;
- damping;
- pretension;
- network topology.

Appropriate parameter selection is required for physically meaningful simulations.

---

# Current Implementation

Implemented:

- Explicit integration
- Force accumulation
- XPBD constraints
- Material lifecycle support

---

# Planned Improvements

Future versions may include:

- Implicit integration
- Adaptive time stepping
- Sparse linear solvers
- Parallel computation
- GPU acceleration

---

# Design Principle

The solver is intentionally independent of biological tissue models.

Its sole responsibility is to advance the mechanical state of the system through time while delegating constitutive behavior to the material subsystem.