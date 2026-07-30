# Material System

## Overview

The material system defines the constitutive mechanical behavior of elements within the ROIF Engine.

The engine separates **geometry**, **network topology**, and **material behavior**, allowing different biological tissues to share the same structural framework while exhibiting different mechanical responses.

---

# Design Philosophy

Each `Element` is responsible only for geometry and kinematics.

Each `Material` is responsible only for constitutive mechanics.

This separation allows new material models to be introduced without modifying the simulation core.

---

# Material Lifecycle

Each simulation step follows the same sequence.

```
begin_step()

↓

force()

↓

end_step()
```

The three-stage lifecycle provides a clean separation between state updates and force computation.

---

# Current Material Models

The current implementation supports:

| Material Property | Status |
|-------------------|:------:|
| Elastic response | ✅ |
| Viscoelastic damping | ✅ |
| Pretension | ✅ |
| Fatigue | ✅ |
| Recovery | ✅ |
| Remodeling | ✅ |
| Failure | ✅ |

---

# Elastic Response

Elastic forces are computed from the current extension relative to the reference length.

The implementation assumes reversible elastic deformation.

---

# Viscoelastic Damping

Velocity-dependent damping reduces oscillations and improves numerical stability.

The current implementation behaves similarly to a Kelvin–Voigt model.

---

# Pretension

Materials may possess an initial internal tension before external loading.

Pretension allows simulation of biologically pre-stressed structures.

---

# Fatigue

Repeated loading accumulates internal damage.

Fatigue reduces the effective mechanical capacity of the material over time.

---

# Recovery

Recovery gradually restores material capacity when loading decreases.

Recovery is modeled independently from fatigue accumulation.

---

# Remodeling

Material parameters may evolve over time in response to mechanical loading.

Current remodeling is simplified and serves as the basis for future mechanobiological extensions.

---

# Failure

Materials can permanently lose their load-bearing capacity.

After failure:

- stiffness becomes zero;
- internal force generation stops;
- the element remains part of the network unless removed explicitly.

---

# Planned Material Models

Future versions are expected to include:

- Maxwell model
- Standard Linear Solid (SLS)
- True creep
- Stress relaxation
- Nonlinear elasticity
- Plastic deformation
- Damage mechanics
- Healing

---

# Extending the Material System

New materials should implement the same lifecycle:

- begin_step()
- force()
- end_step()

This allows integration into the simulation engine without modifications to the solver.

---

# Guiding Principle

The material system is designed to model biological tissues as adaptive mechanical entities whose properties evolve over time while remaining independent of network topology and numerical integration.