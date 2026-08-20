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

The current reproducibility runner version is:

```text
roif_entropy_reproducibility_v2
```

This version verifies not only successful benchmark execution and artifact validity, but also byte-level equality of tracked benchmark artifacts before and after the reproduction run.

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
5. calculates SHA-256 hashes for expected reproducibility artifacts before and after benchmark execution;
6. verifies byte-level equality of tracked benchmark artifacts;
7. classifies artifact outcomes as `REPRODUCED_EXACTLY`, `CHANGED`, `MISSING`, or `NEW`;
8. writes machine-readable and human-readable reproduction reports.

Generated reports:

```text
reproducibility_results/environment.json
reproducibility_results/artifact_snapshot_before.json
reproducibility_results/artifact_snapshot_after.json
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
ARTIFACT EQUALITY: PASS
REPRODUCIBILITY RESULT: PASS
```

This means that the benchmark chain executed successfully and that all tracked expected benchmark artifacts remained byte-identical after regeneration.

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

For tracked expected artifacts, the runner records SHA-256 hashes before and after benchmark execution.

The artifact comparison status is classified as:

```text
REPRODUCED_EXACTLY
CHANGED
MISSING
NEW
```

A tracked expected artifact must finish as `REPRODUCED_EXACTLY` for artifact equality to pass.

If a tracked expected artifact changes, disappears, or otherwise fails exact regeneration, the reproduction run fails even if the corresponding benchmark process itself exits successfully.

## Reproducibility levels

The workflow separates several reproducibility checks that should not be conflated.

### 1. Executable reproducibility

The benchmark scripts must execute successfully from the repository.

### 2. Artifact validity

Expected JSON and CSV outputs must exist and satisfy basic content validation.

JSON outputs must remain parseable and expected artifacts must not be empty.

### 3. Artifact equality

Tracked expected benchmark artifacts are hashed before and after execution.

For exact reproduction, their SHA-256 hashes must remain unchanged.

This detects cases in which a benchmark executes successfully but silently changes a stored manuscript-facing result.

### 4. Focused regression verification

Tests directly associated with manuscript-facing architecture, benchmark generation, claim boundaries, structured memory transitions, predictive preconfiguration, redistribution, and Temporal Image reconstruction are executed as part of the default reproduction path.

### 5. Full repository regression verification

The optional `--full-tests` mode executes the complete repository regression suite after the manuscript-facing benchmark and focused-test chain has passed.

## Failure behavior

The final reproduction result is reported as `FAIL` if:

- an executable benchmark step fails;
- publication asset generation fails;
- focused manuscript tests fail;
- requested full repository tests fail;
- expected artifact content is invalid or missing;
- a tracked expected artifact is not reproduced byte-identically.

The runner therefore distinguishes successful execution from successful reproduction.

A benchmark returning exit code zero is not, by itself, sufficient to establish a successful reproduction.

## Machine-readable evidence

The reproduction workflow generates machine-readable evidence in:

```text
reproducibility_results/reproduction_report.json
```

The report records:

- runner version;
- computational claim scope;
- execution environment;
- repository commit and Git description;
- executed reproduction steps;
- return codes;
- execution durations;
- artifact validity;
- artifact SHA-256 values;
- before/after artifact comparison;
- whether full repository tests were requested.

Environment metadata are also written separately to:

```text
reproducibility_results/environment.json
```

Before/after artifact states are stored in:

```text
reproducibility_results/artifact_snapshot_before.json
reproducibility_results/artifact_snapshot_after.json
```

Individual stdout and stderr logs are written under:

```text
reproducibility_results/logs/
```

These run-specific outputs are generated locally by the reviewer or researcher executing the reproduction workflow.

## Scientific claim boundary

A successful reproduction demonstrates that the published computational paths can be rerun from the repository, that the expected manuscript-facing artifacts can be regenerated, and that tracked expected benchmark artifacts remain byte-identical under the tested reproduction environment.

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

The reproduction workflow verifies the implemented computational experiments and their stored outputs. It does not expand the scientific claims beyond the scope tested by those experiments.

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
Optional Full Regression Suite
      +
Environment Metadata
      +
SHA-256 Artifact Snapshots
      +
Before/After Artifact Equality
      ↓
Independent Reproduction Check
```

The critical distinction is:

```text
benchmark executes successfully
            !=
stored result reproduces exactly
```

The runner therefore checks both.

A successful final reproduction requires:

```text
Benchmark Execution
        ↓
Publication Asset Regeneration
        ↓
Focused Manuscript Regression
        ↓
Optional Full Repository Regression
        ↓
Artifact Content Validation
        ↓
SHA-256 BEFORE / AFTER
        ↓
Tracked Artifacts Byte-Identical
        ↓
ARTIFACT EQUALITY: PASS
        ↓
REPRODUCIBILITY RESULT: PASS
```

This provides a reviewer with one explicit entry point for computational reproduction while preserving the scientific claim boundaries of the individual ROIF experiments.