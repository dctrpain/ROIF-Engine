# Changelog

All notable changes to the ROIF Engine project will be documented in this file.

The format is based on the principles of "Keep a Changelog".

---

## [v0.1.0-alpha] - 2026-07-29

### Added

- Initial project architecture
- Node class
- Element class
- Material abstraction
- Network management
- Solver
- XPBD constraint framework

### Implemented Material Models

- Elastic response
- Viscoelastic damping
- Pretension
- Fatigue
- Recovery
- Remodeling
- Material failure

### Testing

Implemented automated regression tests:

- Single element
- Chain
- Triangle
- Square
- Cross bracing
- Energy conservation
- Damping
- Pretension
- Failure
- Remodeling
- Fatigue
- Recovery
- Viscoelastic behavior

Current status:

- 13 PASSED
- 1 XFAILED (True creep not yet implemented)

### Documentation

Added:

- README.md
- PROJECT_STATUS.md
- requirements.txt
- .gitignore

---

Future versions will continue documenting all significant changes in this file.