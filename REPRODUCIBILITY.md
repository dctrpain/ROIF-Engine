# ROIF Engine — Manuscript Reproducibility

This document describes how to reproduce the computational experiments and manuscript-facing assets associated with the current ROIF study.

## Reproducibility release

The manuscript reproduction workflow is intended to be frozen in the next patch release after `v1.4.0`.

Current validated architecture release:

```text
v1.4.0
580716c13fd4866484cbb651157002123e337b06
```

The final manuscript reproducibility release will include this runner and will receive its own immutable tag.

## Requirements

- Python compatible with the project configuration;
- project dependencies installed;
- repository checkout containing the manuscript-facing benchmark scripts;
- `pytest` available for regression verification.

## Quick reproduction

From the repository root:

```bash
python -m experiments.run_entropy_reproducibility
```

This command:

1. reruns the manuscript-facing benchmark modules;
2. regenerates publication figures and tables;
3. executes focused manuscript regression tests;
4. validates expected JSON/CSV result artifacts;
5. calculates SHA-256 hashes for reproducibility artifacts;
6. writes a machine-readable and human-readable reproduction report.

Generated reports:

```text
reproducibility_results/environment.json
reproducibility_results/reproduction_report.json
reproducibility_results/reproduction_report.md
reproducibility_results/logs/
```

## Full verification

To reproduce the benchmarks and then execute the complete repository regression suite:

```bash
python -m experiments.run_entropy_reproducibility --full-tests
```

A successful run ends with:

```text
REPRODUCIBILITY RESULT: PASS
```

## Manuscript-facing benchmark chain

The runner currently executes:

```text
Q1–Q3 end-to-end benchmark
Q4 matched additive/multiplicative control
Q5 separability benchmark
Q6 perturbation-stability benchmark
Physical-history integration
Matched-state physical-history identifiability
Structured-memory-conditioned matched-state transition
Restricted predictive preconfiguration
Q7 objective-independence audit
Q7 off-nominal transferability audit
Q8 history-conditioned redistribution capacity
Q8 matched-state redistribution
Temporal Image reconstruction
Multilayer Temporal Image trajectory
Publication asset regeneration
Focused manuscript regression tests
Optional full repository regression suite
```

## Expected result artifacts

The runner validates manuscript-facing outputs including:

```text
benchmark_results/physical_history_integration_v1.json
benchmark_results/separability_benchmark_v1.json
benchmark_results/q6_lyapunov_like_v1.json
benchmark_results/matched_state_history_operator_identifiability_v1.json
benchmark_results/memory_conditioned_matched_state_transition_v1.json
benchmark_results/predictive_stabilization_v1.json
benchmark_results/q7_objective_independence_audit_v1.json
benchmark_results/q7_off_nominal_transferability_audit_v2.json
benchmark_results/q8_history_conditioned_redistribution_capacity_v1.json
benchmark_results/q8_history_conditioned_redistribution_matched_state_v2.json
benchmark_results/temporal_image_reconstruction_v1.json
benchmark_results/temporal_image_reconstruction_methods_v1.csv
benchmark_results/temporal_image_reconstruction_slices_v1.csv
benchmark_results/multilayer_temporal_image_trajectory_v1.json
```

## Scientific claim boundary

A successful reproduction demonstrates that the published computational paths can be rerun from the repository and that the expected manuscript-facing artifacts can be regenerated.

It does **not** by itself establish:

- biological or clinical validity;
- a general history-conditioned operator over the complete `SystemImage`;
- complete future Temporal Image prediction;
- general scalar forecasting superiority;
- objective-independent whole-system predictive stabilization;
- biological learning;
- universal stability.

The predictive benchmark is a restricted tested prestress-preconfiguration mechanism.

The Temporal Image benchmark tests reconstruction and trajectory dependence under controlled conditions and should not be interpreted as ordinary whole-system future forecasting.

## Reproducibility philosophy

The runner is designed to make computational claims independently executable rather than dependent on manual transcription.

The reproducibility chain therefore preserves:

```text
Versioned Code
      +
Versioned Benchmark Inputs
      +
Executable Benchmark Scripts
      +
Stored Machine-Readable Outputs
      +
Publication Asset Generator
      +
Focused Regression Tests
      +
Full Regression Option
      +
Environment Metadata
      +
Artifact Hashes
```

This provides a reviewer with one explicit entry point for computational reproduction.
