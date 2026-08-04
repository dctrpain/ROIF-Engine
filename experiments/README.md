# ROIF Engine — Experimental Package

## Release status

**Project stage:** research prototype  
**Validation stage:** controlled synthetic and computational validation  
**Clinical status:** not clinically validated  
**Decision status:** research support only; not a diagnostic or treatment system  
**Release scope:** reproducible experiments for architecture, consistency, robustness, and material-memory verification

This package documents experiments performed at the current development level of the ROIF Engine. The results demonstrate that the implemented computational pipeline behaves consistently under explicitly defined synthetic conditions. They do **not** establish unique real-world causal identification, clinical efficacy, or generalization to arbitrary biological systems.

---

## Development-level scale

The package uses the following internal evidence levels.

| Level | Meaning | Current status |
|---|---|---|
| **L0 — Unit implementation** | Individual classes, invariants, validation, serialization, and numerical safety are tested. | Achieved |
| **L1 — Controlled pipeline consistency** | A known synthetic history is reconstructed from an observed structural signature within a fixed candidate library. | Achieved |
| **L2 — Controlled robustness** | The pipeline is tested under predefined noise and missing-data conditions. | Achieved |
| **L3 — Randomized comparative validation** | Repeated randomized trials, baselines, ablations, confidence intervals, and held-out scenarios. | Planned |
| **L4 — External domain validation** | Validation on independently generated simulations, external datasets, or prospective expert-labelled cases. | Not yet performed |
| **L5 — Clinical validation** | Prospective clinical evaluation with predefined endpoints and governance. | Not yet performed |

The experiments currently released belong primarily to **L1–L2**. The SLS material-memory implementation is verified at **L0**, with system-level regression coverage.

---

## Included experiments

### Example 25 — Inverse structural-history reconstruction

**Purpose:** demonstrate internal consistency of inverse history reconstruction.

**Pipeline:**

```text
known hidden history
→ StructuralSignature
→ competing causal hypotheses
→ forward reconstruction
→ HistoryDecoder ranking
```

**Current evidence level:** **L1 — controlled pipeline consistency**

**Supported claim:** within the explicitly defined candidate library, the decoder can recover the hidden synthetic history whose forward-predicted signature best matches the observation.

**Not supported:** unique reconstruction of arbitrary real-world history.

---

### Example 26 — Robustness under noise and partial observability

**Purpose:** test whether the selected history class remains top-ranked after controlled degradation of the observation.

**Conditions include:**

- moderate profile and capacity noise;
- high noise;
- unavailable agent profile;
- unavailable target profile;
- masked secondary history;
- combined degradation.

**Current evidence level:** **L2 — controlled robustness**

**Supported claim:** the pipeline remains stable under the predefined deterministic perturbation model and fixed candidate library.

**Not supported:** population-level accuracy, calibrated probability of truth, or robustness to unrestricted out-of-distribution inputs.

---

### Example 27 — Unified history-to-action pipeline

**Purpose:** demonstrate the complete orchestration chain.

```text
observed StructuralSignature
→ HypothesisGenerator
→ HistoryDecoder
→ future candidates
→ FuturePlaneAnalyzer
→ HistoryForecaster
→ CounterfactualEngine
→ Node*
```

The experiment also demonstrates candidate-specific future-plane influence and rejection of unsafe system-wide forcing through the **Non-Fonit Gate**.

**Current evidence level:** **L1 — controlled end-to-end consistency**

**Supported claim:** all implemented stages can operate together in one reproducible synthetic experiment.

**Not supported:** guaranteed intervention efficacy or clinical decision validity.

---

### SLS material creep and rheological memory

**Purpose:** verify that material history is represented as an evolving physical state rather than only as an external event record.

Implemented capabilities include:

- delayed creep under sustained load;
- finite relaxed equilibrium;
- stress relaxation under fixed extension;
- recovery after unloading;
- explicit `creep_strain`;
- explicit `rheology_time`;
- snapshot support;
- parameter validation;
- timestep stability;
- finite numerical energy.

**Current evidence level:** **L0 — unit and regression verified**

The implementation passed the full engine regression suite at the time of release:

```text
4441 passed
```

**Supported claim:** the implemented SLS/internal-variable model satisfies the repository's current computational specification without breaking existing tests.

**Not supported:** direct correspondence to a specific biological tissue without parameter identification and external experimental validation.

---

## Reproducibility

Recommended environment:

- Python 3.13
- pytest 9.x
- dependencies defined by the repository configuration

Run the full test suite:

```powershell
python -m pytest -q
```

Run the main demonstrations:

```powershell
python examples\example_25_inverse_history_reconstruction.py
python examples\example_26_inverse_history_robustness.py
python examples\example_27_history_pipeline.py
```

Run the rheology specification:

```powershell
python -m pytest tests\test_creep.py tests\test_material_creep.py -v
```

---

## Interpretation rules

Every published result must be labelled with:

1. the project stage;
2. the evidence level;
3. whether the data are synthetic, simulated, retrospective, or prospective;
4. the candidate-library scope;
5. the perturbation model;
6. the number of trials;
7. the baseline methods;
8. the uncertainty method;
9. the limitations;
10. the exact commit hash.

Avoid statements such as:

- “ROIF proves the true cause”;
- “the decoder identifies real history”;
- “Node* is clinically optimal”;
- “confidence is a probability of historical truth.”

Preferred wording:

> At the current research-prototype stage, the experiment demonstrates computational consistency within an explicitly defined synthetic candidate library.

---

## Current release statement

> These experiments were conducted at the research-prototype stage of the ROIF Engine. The released evidence covers unit verification, controlled synthetic pipeline consistency, and controlled robustness to predefined perturbations. Randomized comparative validation, external validation, and clinical validation have not yet been completed.

---

## Next validation package — L3

The next publication package should add:

- randomized scenario generation;
- train/development/held-out separation;
- nearest-signature baseline;
- prior-only baseline;
- additive-versus-multiplicative comparison;
- ablation without history reconstruction;
- ablation without future planes;
- ablation without the Non-Fonit Gate;
- top-1 and top-k recovery;
- confidence intervals;
- calibration analysis;
- runtime and memory benchmarks;
- machine-readable result files.

Until that package is completed, the current release should remain labelled **controlled synthetic validation of a research prototype**.
