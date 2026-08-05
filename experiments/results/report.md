# ROIF Experimental Package Report

## Release status

- **Project stage:** research prototype
- **Validation scope:** controlled synthetic and computational validation
- **Clinical validation:** not performed
- **Package version:** 1.1.0
- **Reference milestone:** rheological memory integrated into structural history
- **Commit:** `74affb360d1926f331ff2202699401cc507495d6`
- **Generated:** `2026-08-04T11:14:25.775727+00:00`

## Summary

- Experiments: **6**
- Passed: **6**
- Failed: **0**

| ID | Level | Data | Status | Duration, s |
|---|---:|---|---|---:|
| `example_25` | L1 | synthetic | **PASSED** | 0.450 |
| `example_26` | L2 | synthetic | **PASSED** | 0.474 |
| `example_27` | L1 | synthetic | **PASSED** | 0.481 |
| `example_28` | L1 | synthetic_computational | **PASSED** | 0.471 |
| `sls_material` | L0 | computational | **PASSED** | 61.677 |
| `rheological_history_contract` | L0 | computational | **PASSED** | 1.139 |

## Interpretation

These results demonstrate computational behaviour at the research-prototype stage. They do not establish unique real-world causal identification, clinical efficacy, or generalization to arbitrary biological systems.

## Experiment details

### example_25 — Inverse structural-history reconstruction

- Level: **L1**
- Data type: **synthetic**
- Status: **passed**
- Return code: `0`
- Duration: `0.449553 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe examples/example_25_inverse_history_reconstruction.py`
- Stdout: `experiments/results/logs/example_25.stdout.txt`
- Stderr: `experiments/results/logs/example_25.stderr.txt`

### example_26 — Robustness under noise and partial observability

- Level: **L2**
- Data type: **synthetic**
- Status: **passed**
- Return code: `0`
- Duration: `0.474413 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe examples/example_26_inverse_history_robustness.py`
- Stdout: `experiments/results/logs/example_26.stdout.txt`
- Stderr: `experiments/results/logs/example_26.stderr.txt`

### example_27 — Unified history-to-action pipeline

- Level: **L1**
- Data type: **synthetic**
- Status: **passed**
- Return code: `0`
- Duration: `0.481064 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe examples/example_27_history_pipeline.py`
- Stdout: `experiments/results/logs/example_27.stdout.txt`
- Stderr: `experiments/results/logs/example_27.stderr.txt`

### example_28 — Rheological memory becomes structural history

- Level: **L1**
- Data type: **synthetic_computational**
- Status: **passed**
- Return code: `0`
- Duration: `0.470763 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe -m examples.example_28_rheological_memory`
- Stdout: `experiments/results/logs/example_28.stdout.txt`
- Stderr: `experiments/results/logs/example_28.stderr.txt`

### sls_material — SLS material creep and rheological memory

- Level: **L0**
- Data type: **computational**
- Status: **passed**
- Return code: `0`
- Duration: `61.677341 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe -m pytest tests/test_creep.py tests/test_material_creep.py -q`
- Stdout: `experiments/results/logs/sls_material.stdout.txt`
- Stderr: `experiments/results/logs/sls_material.stderr.txt`

### rheological_history_contract — Rheological memory integration contract

- Level: **L0**
- Data type: **computational**
- Status: **passed**
- Return code: `0`
- Duration: `1.139053 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe -m pytest tests/test_rheology_memory.py tests/test_irreversible_change.py tests/test_structural_signature.py -q`
- Stdout: `experiments/results/logs/rheological_history_contract.stdout.txt`
- Stderr: `experiments/results/logs/rheological_history_contract.stderr.txt`

## Required publication label

> Controlled synthetic and computational validation of a research prototype. Randomized comparative, external, and clinical validation have not yet been completed.
