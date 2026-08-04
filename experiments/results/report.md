# ROIF Experimental Package Report

## Release status

- **Project stage:** research prototype
- **Validation scope:** controlled synthetic and computational validation
- **Clinical validation:** not performed
- **Commit:** `c7948e402c8de5319ad4130edda9f01e33fc279d`
- **Generated:** `2026-08-04T07:37:53.974976+00:00`

## Summary

- Experiments: **4**
- Passed: **4**
- Failed: **0**

| ID | Level | Data | Status | Duration, s |
|---|---:|---|---|---:|
| `example_25` | L1 | synthetic | **PASSED** | 1.163 |
| `example_26` | L2 | synthetic | **PASSED** | 1.100 |
| `example_27` | L1 | synthetic | **PASSED** | 1.570 |
| `sls_material` | L0 | computational | **PASSED** | 86.118 |

## Interpretation

These results demonstrate computational behaviour at the research-prototype stage. They do not establish unique real-world causal identification, clinical efficacy, or generalization to arbitrary biological systems.

## Experiment details

### example_25 — Inverse structural-history reconstruction

- Level: **L1**
- Data type: **synthetic**
- Status: **passed**
- Return code: `0`
- Duration: `1.163104 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe examples/example_25_inverse_history_reconstruction.py`
- Stdout: `experiments/results/logs/example_25.stdout.txt`
- Stderr: `experiments/results/logs/example_25.stderr.txt`

### example_26 — Robustness under noise and partial observability

- Level: **L2**
- Data type: **synthetic**
- Status: **passed**
- Return code: `0`
- Duration: `1.100467 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe examples/example_26_inverse_history_robustness.py`
- Stdout: `experiments/results/logs/example_26.stdout.txt`
- Stderr: `experiments/results/logs/example_26.stderr.txt`

### example_27 — Unified history-to-action pipeline

- Level: **L1**
- Data type: **synthetic**
- Status: **passed**
- Return code: `0`
- Duration: `1.570114 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe examples/example_27_history_pipeline.py`
- Stdout: `experiments/results/logs/example_27.stdout.txt`
- Stderr: `experiments/results/logs/example_27.stderr.txt`

### sls_material — SLS material creep and rheological memory

- Level: **L0**
- Data type: **computational**
- Status: **passed**
- Return code: `0`
- Duration: `86.118318 s`
- Command: `C:\Users\DELL\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe -m pytest tests/test_creep.py tests/test_material_creep.py -q`
- Stdout: `experiments/results/logs/sls_material.stdout.txt`
- Stderr: `experiments/results/logs/sls_material.stderr.txt`

## Required publication label

> Controlled synthetic validation of a research prototype. Randomized comparative, external, and clinical validation have not yet been completed.
