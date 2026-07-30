# ROIF Engine Architecture

## Overview

ROIF Engine is a modular biomechanical simulation engine designed to model adaptive pre-stressed biological systems.

The architecture emphasizes:

- modularity;
- extensibility;
- deterministic simulation;
- scientific reproducibility;
- separation of mechanics from material behavior.

---

# High-Level Architecture

```
                 +----------------------+
                 |      Simulation      |
                 +----------+-----------+
                            |
                     +------+------+
                     |   Network    |
                     +------+------+
                            |
          +-----------------+-----------------+
          |                                   |
      +---+---+                           +---+---+
      | Node  |                           |Element|
      +-------+                           +---+---+
                                              |
                                        +-----+------+
                                        |  Material  |
                                        +-----+------+
                                              |
                 +----------------------------+----------------------------+
                 |             |              |            |               |
              Muscle        Tendon        Ligament      Fascia      Future Models
```

---

# Core Components

## Node

Represents a point in space.

Stores:

- position
- velocity
- accumulated forces
- mass
- fixed/free state

Nodes contain no constitutive behavior.

---

## Element

Represents a mechanical connection between two nodes.

Responsibilities:

- compute current length
- compute extension
- compute strain
- request material forces
- apply forces to connected nodes

Elements do not define constitutive equations.

---

## Material

Defines constitutive mechanical behavior.

Responsibilities:

- elastic response
- damping
- fatigue
- recovery
- remodeling
- failure

Material models are independent from network topology.

---

## Network

Represents the complete biomechanical system.

Responsibilities:

- node management
- element management
- simulation stepping
- force accumulation
- solver interaction

Network owns the global simulation state.

---

## Solver

Responsible for numerical integration.

Current implementation:

- explicit integration
- XPBD constraints

Future versions may include:

- implicit solvers
- adaptive time stepping
- sparse solvers

---

# Design Principles

## Separation of Responsibilities

Each class has a single primary responsibility.

| Class | Responsibility |
|--------|----------------|
| Node | State |
| Element | Mechanics |
| Material | Constitutive behavior |
| Network | Simulation |
| Solver | Numerical integration |

---

## Material Independence

Mechanical behavior is encapsulated inside Material classes.

This allows different constitutive models to be attached to the same network topology without modifying the solver.

---

## Extensibility

New materials can be added by implementing the Material interface.

Examples include:

- Muscle
- Tendon
- Fascia
- Ligament
- Cartilage
- Bone

---

## Testing Strategy

Every new feature must be accompanied by automated tests.

Regression testing is considered a core architectural principle.

---

# Current Architecture

Implemented:

- Node
- Element
- Material
- Network
- Solver
- XPBD constraints

Current status:

Stable for alpha development.

---

# Future Evolution

The architecture is intentionally modular to support:

- active muscle mechanics;
- mechanobiology;
- tissue growth;
- healing;
- patient-specific simulations;
- whole-body biomechanical models.

---

# Guiding Principle

The engine separates **structure**, **mechanics**, and **material behavior**, allowing biological complexity to emerge from modular interactions rather than being hard-coded into the simulation core.