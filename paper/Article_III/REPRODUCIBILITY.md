# Article III — Computational Reproducibility

## Purpose

This document describes the computational reproduction procedure for the frozen scientific results used in Article III.

The reproduction layer is deliberately separated from the scientific freeze. The scientific benchmarks, benchmark results, regression tests, and available pre-specified protocol artifacts were frozen before the reviewer-facing orchestration runner was added.

The runner does not implement an alternative scientific model and does not replace the benchmark-specific tests. It calls the frozen benchmark implementations and executes their frozen regression tests.

## 1. Scientific Freeze

Scientific Freeze 0:

    6dd29ca5fe529f6a285296e08a5e87be98d81f99

Commit:

    science: freeze Article III falsification suite

Freeze manifest:

    paper/Article_III/SCIENTIFIC_FREEZE_0_SHA256.txt

The manifest records SHA-256 hashes of the frozen scientific assets.

The freeze establishes repository provenance from that point forward. It must not be interpreted as evidence that every protocol artifact historically predated every exploratory or preliminary computation.

## 2. Post-Freeze Reproduction Infrastructure

The reviewer-facing runner was added later in a separate commit:

    e26602ad7f4560adf6f8250ed00ced6416ac9d71

Commit:

    reproducibility: add Article III manuscript suite runner

Runner:

    experiments/run_manuscript_suite.py

The runner is orchestration infrastructure. It does not modify the registered experimental conditions or introduce an alternative implementation of the tested mechanisms.

## 3. One-Command Reproduction

From the repository root:

    python -m experiments.run_manuscript_suite

Successful reproduction:

    Mechanism decomposition................... PASS
    Retained prestress........................ PASS
    Z7 reduction falsification................ PASS
    Local response-rank audit................. PASS

    43 passed
    Frozen regression suite................... PASS

    REGISTERED FALSIFICATION SUITE REPRODUCED

Execution time is environment-dependent and is not part of the scientific claim.

## 4. Computational Claim 1 — Mechanism Decomposition

Benchmark:

    experiments/transition_order_mechanism_falsification.py

Frozen result:

    benchmark_results/transition_order_mechanism_falsification.json

Regression test:

    tests/test_transition_order_mechanism_falsification.py

Frozen result SHA-256:

    aab5ede2d24626ac016053c3b1a1cd6519ec2d2c8ce19902f0edbdebb4a9f8e6

The registered comparison separates three regimes:

    I   additive reference
    II  fixed-network production redistribution
    III reconfiguring production network

Frozen registered distances:

    D_additive = 0
    D_fixed = 2.77555756e-17
    D_reconfiguring = 0.011814192037381752

Registered numerical tolerance:

    1e-12

The registered result supports the limited computational statement that order dependence is absent within tolerance in the additive and fixed controls but is present in the tested reconfiguring regime.

It does not establish that all nonlinear systems require reconfiguration, that all reconfiguring systems are order-dependent, or that order dependence is unique to this implementation.

Registration status:

The benchmark contains embedded pre-specified conditions and decision logic. No separate independent preregistration artifact is claimed for this benchmark.

## 5. Computational Claim 2 — Retained Prestress

Pre-specified protocol artifact:

    experiments/retained_prestress_future_transition_preregistration.json

Benchmark:

    experiments/retained_prestress_future_transition_falsification.py

Frozen result:

    benchmark_results/retained_prestress_future_transition_falsification.json

Regression test:

    tests/test_retained_prestress_future_transition_falsification.py

Frozen result SHA-256:

    e742f28a725022dae1c8897d8b497091a530abfc1998a988b444c8b8550b736e

The registered AB and BA sequences reach equivalent terminal adaptive connection states while retaining distinguishable distributed prestress:

    terminal connection distance = 0
    terminal prestress distance = 0.011814192037381752

After the registered locked READ:

    post-READ connection distance = 0
    post-READ prestress distance = 0.011814192037381738

Registered decision:

    RETAINED_PRESTRESS_IS_TRANSITION_RELEVANT_FOR_REGISTERED_READ

This supports the limited conclusion that, in the registered regime, removing the retained prestress coordinates while retaining the terminal adaptive connection state loses information relevant to the subsequent registered transition.

It does not establish that prestress is universally irreducible.

## 6. Computational Claim 3 — Z7 Reduction Falsification

Benchmark:

    experiments/transition_sufficient_z7_falsification.py

Frozen result:

    benchmark_results/transition_sufficient_z7_falsification.json

Regression test:

    tests/test_transition_sufficient_z7_falsification.py

Frozen result SHA-256:

    c571c867867cff27efb79776df11c4cf97592b1c840a8f34b7be0e75faeb4b79

The tested reduced representation is:

    Z7 = (P_A, P_B, P_C, g_AB, c_AB, g_BC, c_BC)

where:

    g = s * c * r * (1 - f)

Two registered source states collide exactly in Z7 while differing in their underlying adaptive decomposition.

For the restricted direct mechanical probe, the reduction remains equivalent within tolerance.

After an identical registered adaptive WRITE, the reduced states diverge:

    Z7 distance after WRITE = 0.033213859765625076

After the identical registered future READ, the physical outputs diverge:

    future prestress distance = 0.0077844983825683944

Registered decision:

    Z7_NOT_TRANSITION_SUFFICIENT_OVER_REGISTERED_DOMAIN

The result falsifies transition sufficiency of this specific Z7 reduction over the registered event domain.

It does not establish that every lower-dimensional representation fails, that the full registered state is globally minimal, or that an explicit historical record is always required.

Registration status:

The benchmark contains embedded pre-specified null, event domain, tolerance, controls, and decision logic. No separate independent preregistration artifact is claimed for this benchmark.

## 7. Computational Claim 4 — Local Future-Response Rank

Pre-specified protocol artifact:

    experiments/transition_sufficient_local_rank_preregistration.json

Benchmark:

    experiments/transition_sufficient_local_rank_falsification.py

Frozen result:

    benchmark_results/transition_sufficient_local_rank_falsification.json

Regression test:

    tests/test_transition_sufficient_local_rank_falsification.py

Frozen result SHA-256:

    a9e800edf10bd7f2ee1ada390327a687b7c4a9ed8095eeebb9c56ed5dcba1bfb

The registered adaptive connection state is:

    C = (s, c, r, f, b)

The registered future-response map concatenates effective transfer gain and contractile capacity after the registered probe set.

The numerical Jacobian was evaluated using central finite differences at:

    h = 1e-5
    h = 1e-6
    h = 1e-7

The registered numerical rank was:

    rank = 5

at all three step sizes.

At the primary step size h = 1e-6, the singular values were:

    4.4325908460411245
    1.870950186764567
    0.062405351103277794
    0.03457268722507757
    0.016806162529913963

Registered decision:

    LOCAL_SMOOTH_EXACT_REDUCTION_BELOW_5D_EXCLUDED

This is a numerical certificate for the registered response map and registered neighborhood.

It supports a local lower bound against smooth exact reductions below five dimensions under the registered conditions.

It is not a theorem of global minimality and does not exclude non-smooth, discontinuous, task-specific, or differently defined representations.

## 8. Regression Layer

The reviewer-facing runner executes the four frozen regression modules:

    tests/test_transition_order_mechanism_falsification.py
    tests/test_retained_prestress_future_transition_falsification.py
    tests/test_transition_sufficient_z7_falsification.py
    tests/test_transition_sufficient_local_rank_falsification.py

At the post-freeze reproduction check, the combined result was:

    43 passed

The four regenerated machine-readable benchmark result files matched their Scientific Freeze 0 SHA-256 values exactly.

A post-reproduction Git integrity audit showed no tracked changes in the frozen benchmark scripts, frozen result JSON files, or frozen regression tests.

## 9. Registration and Provenance Boundary

The four computational lines do not have identical registration provenance.

Separate pre-specified protocol artifacts exist for:

    retained prestress
    local future-response rank

For:

    mechanism decomposition
    Z7 reduction falsification

the benchmark source contains embedded pre-specified conditions and decision logic, but no separate independent preregistration artifact is claimed.

Accordingly, the manuscript must not describe all four experiments as "preregistered experiments."

The conservative description of the complete set is:

    pre-specified computational falsification suite

with the registration status of individual benchmarks reported separately.

Scientific Freeze 0 establishes repository provenance from commit:

    6dd29ca5fe529f6a285296e08a5e87be98d81f99

It does not retrospectively establish historical preregistration chronology.

## 10. Claim Boundaries

Successful execution of this suite does NOT establish:

    Biological validity.
    Clinical validity.
    Global state minimality.
    Universal irreducibility of prestress.
    Failure of every possible state reduction.
    Universal necessity of nonlinear reconfiguration.
    Uniqueness of order dependence to this model.
    Uniqueness of the underlying mathematical concepts.

Established concepts from observability, minimal realization, bisimulation, lumpability, predictive-state representations, state abstraction, hysteresis, and path-dependent dynamics are not claimed as new mathematical theory by this reproduction suite.

The computational contribution must therefore be interpreted at the level of the explicitly registered mechanisms, reductions, event domains, observables, and numerical tolerances.

## 11. Reproduction Invariant

For every central computational statement used in Article III, the intended audit chain is:

    manuscript claim
          |
          v
    pre-specified computational condition
          |
          v
    executable frozen benchmark
          |
          v
    machine-readable result
          |
          v
    benchmark-specific regression test
          |
          v
    reviewer-facing reproduction suite

A manuscript claim must not be stronger than the frozen artifact that supports it.

## 12. Current Reproduction Status

Scientific freeze:

    6dd29ca5fe529f6a285296e08a5e87be98d81f99

Reviewer-runner commit:

    e26602ad7f4560adf6f8250ed00ced6416ac9d71

Current registered reproduction status:

    Mechanism decomposition ............. PASS
    Retained prestress .................. PASS
    Z7 reduction falsification .......... PASS
    Local response-rank audit ........... PASS

    Regression tests .................... 43/43 PASS

    Frozen result regeneration .......... EXACT
    Post-reproduction frozen diff ....... CLEAN

This document records computational reproducibility. It does not by itself establish novelty, biological interpretation, clinical applicability, or general theoretical validity.
