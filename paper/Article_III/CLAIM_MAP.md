# Article III — Claim Map

## Purpose

This document controls the scientific claims of Article III.

It separates:

1. established theory that must not be presented as novel;
2. computational results demonstrated by the frozen Article III suite;
3. interpretations directly supported by those results;
4. stronger statements that the manuscript is not allowed to make.

This is an internal scientific control document, not manuscript prose.

The governing rule is:

    A manuscript claim must not be stronger than the frozen computational
    artifact and established theory that support it.

Scientific Freeze 0:

    6dd29ca5fe529f6a285296e08a5e87be98d81f99

Reviewer reproduction documentation:

    paper/Article_III/REPRODUCIBILITY.md

---

## 1. Central Scientific Question

Registered state:

    I_t = (P_t, C_t)

where:

    P_t = distributed prestress state
    C_t = adaptive connection state

Registered transition:

    I_(t+1) = T(I_t, e_t; Gamma)

For a reduced representation:

    Z_t = Pi(I_t)

the central question is:

Can a state reduction that is exact for the current propagation regime cease
to be transition-sufficient when admissible events can reconfigure the
operator through which subsequent perturbations propagate?

This question must remain narrower than any universal claim about memory,
history dependence, nonlinear systems, adaptive networks, or state reduction.

---

## 2. Established Theory — Not Our Novelty

The following conceptual or mathematical ideas are treated as established
background and must not be claimed as discoveries of Article III:

    state equivalence under future transitions;
    quotient closure;
    bisimulation;
    lumpability;
    projectability;
    observability;
    nonlinear observability;
    minimal realization;
    sufficient-state representations;
    predictive-state representations;
    action- or event-conditioned distinguishability;
    path dependence;
    hysteresis;
    noncommuting transition sequences;
    adaptive and co-evolving networks;
    prestress-dependent mechanical response.

Article III may use these concepts to define tests or interpret results.

Their general mathematical existence is not a contribution of this work.

---

## 3. Claim 1 — Reconfiguration Introduces Registered Order Dependence

### Candidate manuscript claim

In the registered three-node system, the tested AB/BA perturbation sequences
are equivalent within numerical tolerance in the additive reference and
fixed-network redistribution regimes, but become distinguishable when the
registered adaptive reconfiguration mechanism is enabled.

### Established theory

Order dependence, path dependence, hysteresis, and noncommutativity are
established phenomena.

Therefore:

    AB != BA

is not itself a novelty claim.

### Our frozen evidence

Benchmark:

    experiments/transition_order_mechanism_falsification.py

Frozen result:

    benchmark_results/transition_order_mechanism_falsification.json

Regression test:

    tests/test_transition_order_mechanism_falsification.py

Registered results:

    D_additive      = 0
    D_fixed         = 2.77555756e-17
    D_reconfiguring = 0.011814192037381752

Registered tolerance:

    1e-12

Frozen result SHA-256:

    aab5ede2d24626ac016053c3b1a1cd6519ec2d2c8ce19902f0edbdebb4a9f8e6

### Supported interpretation

Within the registered benchmark, enabling the tested reconfiguration
mechanism changes the system from an AB/BA-equivalent regime to an
AB/BA-distinguishable regime.

This identifies a concrete mechanism responsible for the observed order
dependence in this implementation.

### Forbidden stronger claims

Do NOT write:

    Reconfiguration generally causes noncommutativity.

    Nonlinear systems require reconfiguration to become path-dependent.

    Fixed networks are generally order-independent.

    Order dependence is unique to this framework.

    The benchmark proves a universal property of adaptive networks.

### Status

    SUPPORTED WITH REGISTERED SCOPE

---

## 4. Claim 2 — Retained Prestress Is Transition-Relevant

### Candidate manuscript claim

In the registered reconfiguring network, two perturbation sequences can reach
the same terminal adaptive connection state while retaining different
distributed prestress states, and that prestress difference remains relevant
to a subsequent registered transition.

### Established theory

Prestress affects mechanical response, and physical systems can retain
path-dependent internal state.

Neither statement is claimed as new.

### Our frozen evidence

Pre-specified protocol artifact:

    experiments/retained_prestress_future_transition_preregistration.json

Benchmark:

    experiments/retained_prestress_future_transition_falsification.py

Frozen result:

    benchmark_results/retained_prestress_future_transition_falsification.json

Regression test:

    tests/test_retained_prestress_future_transition_falsification.py

Registered terminal state:

    terminal connection distance = 0

    terminal prestress distance =
        0.011814192037381752

After the registered locked READ:

    post-READ connection distance = 0

    post-READ prestress distance =
        0.011814192037381738

Registered decision:

    RETAINED_PRESTRESS_IS_TRANSITION_RELEVANT_FOR_REGISTERED_READ

Frozen result SHA-256:

    e742f28a725022dae1c8897d8b497091a530abfc1998a988b444c8b8550b736e

### Supported interpretation

For the registered transition domain:

    (P, C) -> C

is not a transition-sufficient reduction.

The preceding sequence can therefore remain physically encoded in the
present distributed prestress state even when the registered adaptive
connection state is identical.

The manuscript may state:

    history may be physically encoded in the present state

provided that this statement is explicitly tied to the demonstrated
prestress mechanism and is not generalized to all dynamical systems.

### Forbidden stronger claims

Do NOT write:

    Prestress is universally irreducible.

    Prestress is a new form of memory.

    Every history-dependent system must explicitly retain prestress.

    The full past trajectory can be reconstructed from present prestress.

    Different prestress states always produce different future behavior.

### Status

    SUPPORTED WITH REGISTERED SCOPE

---

## 5. Claim 3 — A Currently Exact Reduction Can Fail Under Reconfiguration

### Candidate manuscript claim

A reduced representation can preserve the registered current mechanical
response exactly while failing to remain transition-sufficient when the
admissible event domain includes adaptive reconfiguration.

### Established theory

State abstraction, sufficient-state representations, bisimulation,
lumpability, predictive-state representations, and event-conditioned
equivalence are established concepts.

The general fact that a quotient must preserve the relevant transition
structure is not claimed as new.

### Our frozen evidence

Tested reduction:

    Z7 = (P_A, P_B, P_C, g_AB, c_AB, g_BC, c_BC)

with:

    g = s * c * r * (1 - f)

Benchmark:

    experiments/transition_sufficient_z7_falsification.py

Frozen result:

    benchmark_results/transition_sufficient_z7_falsification.json

Regression test:

    tests/test_transition_sufficient_z7_falsification.py

The two registered source states differ in their underlying adaptive
decomposition but collide exactly in Z7:

    full-state distance = 0.3201562118716424
    initial Z7 distance = 0

For the restricted direct mechanical probe:

    output distance = 0

Thus Z7 is not rejected by the direct mechanical probe alone.

After the identical registered adaptive WRITE:

    Z7 distance after WRITE =
        0.033213859765625076

After the identical registered future READ:

    future prestress distance =
        0.0077844983825683944

Registered tolerance:

    1e-12

Negative repeat controls:

    distance = 0

Registered decision:

    Z7_NOT_TRANSITION_SUFFICIENT_OVER_REGISTERED_DOMAIN

Frozen result SHA-256:

    c571c867867cff27efb79776df11c4cf97592b1c840a8f34b7be0e75faeb4b79

### Supported interpretation

Z7 is sufficient for the tested restricted direct mechanical probe but is
not closed under the larger registered event domain containing adaptive
WRITE followed by future READ.

The demonstrated mechanism is:

    different internal adaptive decompositions
        ->
    identical current reduced mechanical interface
        ->
    identical adaptive WRITE
        ->
    different updated reduced states
        ->
    different subsequent mechanical response

The important result is therefore not merely that information was removed.

The result shows that information which is behaviorally silent under one
registered event class can become transition-relevant when the admissible
event class is enlarged to include reconfiguration.

### Forbidden stronger claims

Do NOT write:

    Reduced states are generally insufficient.

    Every hidden variable must be retained.

    Z7 failure proves that the full registered state is minimal.

    No lower-dimensional exact representation exists.

    Explicit historical memory is necessary.

    Current behavioral equivalence is always misleading.

    Reconfiguration destroys every valid state reduction.

### Status

    SUPPORTED WITH REGISTERED SCOPE

---

## 6. Claim 4 — Local Smooth Exact Reduction Below Five Dimensions Is Excluded

### Candidate manuscript claim

For the registered adaptive connection response map and the registered
interior neighborhood, the numerical response Jacobian has stable rank five
across the pre-specified finite-difference step sizes.

Under the smooth exact-reduction assumptions of the registered audit, this
excludes a local exact representation below five dimensions.

### Established theory

Observability rank arguments, local distinguishability, differential
embeddings, and minimal-realization reasoning are established mathematical
tools.

The inference from full local differential rank to a local dimensional lower
bound is not claimed as a new theorem.

### Our frozen evidence

Registered adaptive state:

    C = (s, c, r, f, b)

Pre-specified protocol artifact:

    experiments/transition_sufficient_local_rank_preregistration.json

Benchmark:

    experiments/transition_sufficient_local_rank_falsification.py

Frozen result:

    benchmark_results/transition_sufficient_local_rank_falsification.json

Regression test:

    tests/test_transition_sufficient_local_rank_falsification.py

Registered response dimension:

    12

Registered finite-difference step sizes:

    h = 1e-5
    h = 1e-6
    h = 1e-7

Observed numerical rank:

    rank = 5 at h = 1e-5
    rank = 5 at h = 1e-6
    rank = 5 at h = 1e-7

Primary singular values at h = 1e-6:

    4.4325908460411245
    1.870950186764567
    0.062405351103277794
    0.03457268722507757
    0.016806162529913963

Primary relative rank threshold:

    1e-9

Approximate condition number:

    263.7479459

Registered decision:

    LOCAL_SMOOTH_EXACT_REDUCTION_BELOW_5D_EXCLUDED

Frozen result SHA-256:

    a9e800edf10bd7f2ee1ada390327a687b7c4a9ed8095eeebb9c56ed5dcba1bfb

### Supported interpretation

For the registered response map near the registered interior state, all five
adaptive coordinates contribute independent local response directions at the
registered numerical resolution.

This provides a numerical local certificate against smooth exact reductions
below five dimensions for this response map.

### Forbidden stronger claims

Do NOT write:

    Five dimensions are globally minimal.

    Every exact representation requires five explicit physical variables.

    Non-smooth reductions are impossible.

    The rank audit proves global observability.

    The result establishes a new minimal-realization theorem.

    The five-dimensional result applies to the complete ROIF architecture.

### Status

    SUPPORTED WITH REGISTERED SCOPE

---

## 7. Integrated Contribution of Article III

The four frozen computational results must not be presented as four unrelated
demonstrations.

Together they establish one bounded mechanistic chain within the registered
system.

### Step 1 — Identify the source of order sensitivity

The mechanism-decomposition benchmark shows:

    additive regime:
        AB and BA equivalent within tolerance

    fixed-network redistribution:
        AB and BA equivalent within tolerance

    adaptive reconfiguration enabled:
        AB and BA distinguishable

Thus the registered order sensitivity is associated with the tested
reconfiguration mechanism rather than with the additive or fixed-network
controls.

### Step 2 — Identify where sequence information remains physically represented

The retained-prestress benchmark shows:

    C_AB = C_BA

while:

    P_AB != P_BA

and the difference in P remains relevant under the subsequent registered
locked READ.

Thus, in this benchmark, sequence dependence need not be represented by an
explicit historical log.

A distinction generated by the preceding sequence can remain encoded in the
present physical state.

### Step 3 — Show failure of a reduction that is exact for the current probe

The Z7 benchmark constructs two different adaptive states satisfying:

    Z7_A = Z7_B

and:

    direct_READ_A = direct_READ_B

The reduction is therefore not rejected by the registered direct mechanical
probe.

However, after the same admissible adaptive WRITE:

    Z7_A' != Z7_B'

and after the same subsequent READ:

    future_response_A != future_response_B

Thus current behavioral equivalence does not guarantee closure of the same
reduction under an enlarged event domain containing reconfiguration.

### Step 4 — Test whether the adaptive state can be smoothly compressed further

The local future-response rank audit finds stable numerical rank five for the
registered five-dimensional adaptive state across all pre-specified
finite-difference step sizes.

Using established local observability/minimal-realization reasoning, this
provides a numerical local lower bound against smooth exact reductions below
five dimensions for the registered response map.

### Integrated mechanistic chain

The computational contribution of Article III is therefore represented as:

    perturbation order
        ->
    transient reconfiguration
        ->
    altered propagation through the evolving system
        ->
    different retained distributed prestress
        ->
    current physical state carries sequence-dependent information

and, independently but compatibly:

    different latent adaptive decompositions
        ->
    identical current mechanical interface
        ->
    identical admissible WRITE
        ->
    different updated reduced states
        ->
    different future mechanical response

The common principle is:

    A distinction may be safely removed only if removing it does not change
    the admissible subsequent transitions over the event domain and horizon
    for which the representation is intended.

This principle is used as a control rule for the computational study.

Its general mathematical foundations are established and are not claimed as
a new theorem.

---

## 8. Physical Interpretation Boundary

The computational object of Article III is a reconfiguring prestressed
network.

The registered state is:

    I = (P, C)

where P and C have distinct physical roles.

P represents the current distributed prestress state.

C represents the current adaptive connection state.

The article must preserve the following physical causal sequence:

    perturbation
        ->
    redistribution through the current prestressed network
        ->
    adaptive reconfiguration
        ->
    changed physical state
        ->
    changed response to a subsequent perturbation

Mathematical abstractions must not replace this physical object.

For every abstraction used in the manuscript, the following questions must
remain answerable:

    1. What physical quantity does the state represent?

    2. What does the projection remove?

    3. What physical process implements the transition?

    4. What physically relevant distinction could be lost by the reduction?

If these questions cannot be answered, the abstraction is too detached from
the demonstrated mechanism for the central argument of Article III.

---

## 9. Historical Cause Versus Present Transition-Relevant State

Article III must distinguish between:

    historical initiating sequence

and:

    present state that determines subsequent transitions

The computational results do not show that the complete historical sequence
must be stored or reconstructed.

Instead, they show that effects of preceding events can remain represented in
the current physical state.

Therefore the preferred formulation is:

    history may be physically encoded in the present state

rather than:

    the system requires explicit memory of its history

or:

    the complete past determines the future independently of present state.

The manuscript concerns future-relevant present state, not reconstruction of
the complete past trajectory.

---

## 10. Relationship to the Original Physical Architecture

The broader physical architecture motivating the work can be represented as:

    heterogeneous influences
        ->
    evolving material state
        ->
    prestressed configuration
        ->
    redistribution
        ->
    adaptation
        ->
    changed subsequent response

Article III tests only a controlled subset of this architecture.

The frozen experiments do not directly test:

    temperature fields;
    hydration fields;
    chemical composition;
    biological remodeling;
    aging;
    tissue-specific constitutive laws;
    clinical cascades;
    anatomical functional planes;
    full heterogeneous environmental coupling.

These broader elements must not be imported into the Results as if they had
been experimentally demonstrated by the frozen suite.

They may appear only as motivation, architectural context, limitations, or
future testable extensions when clearly identified as such.

---

## 11. Active Probing Boundary

Active probing is a natural extension of the demonstrated distinguishability
problem.

A future probe-selection problem may be written conceptually as:

    choose e from E

to maximize:

    distance(
        O[T(I_A, e)],
        O[T(I_B, e)]
    )

subject to an admissibility or safety constraint on the perturbation.

Article III does not currently demonstrate an optimized non-destructive probe
selection algorithm.

Therefore the manuscript may identify active probe selection as a future
extension but must not claim that the frozen suite provides a general
diagnostic probing method.

---

## 12. Prohibited Central Claims

The following statements are outside the demonstrated scope and must not
appear as central claims in the Title, Abstract, Results, Discussion, or
Conclusion:

    ROIF proves that complex systems require memory.

    History cannot be reduced.

    Reduced representations are fundamentally insufficient.

    Prestress is universally irreducible.

    Reconfiguration always destroys state reductions.

    Nonlinearity necessarily produces path dependence.

    The full registered state is globally minimal.

    Five dimensions are universally necessary.

    All hidden variables must be retained.

    The framework reconstructs the true historical cause.

    The experiments establish biological validity.

    The experiments establish clinical validity.

    The experiments establish a universal theory of complex systems.

    Bisimulation, observability, predictive states, or minimal realization
    are introduced here as new mathematical concepts.

---

## 13. Preferred Central Claim

The current preferred bounded formulation is:

    In the registered prestressed network, transient reconfiguration changes
    the physical state through which subsequent perturbations propagate.

    Consequently, two states that are mechanically indistinguishable under
    the current propagation probe can become distinguishable after the same
    admissible reconfiguration event.

The frozen computational suite demonstrates two concrete physical routes by
which this future-relevant distinction is retained or exposed:

    1. sequence-dependent distributed prestress retained in the present
       physical state;

    2. latent adaptive decomposition that is silent at the current mechanical
       interface but becomes mechanically relevant after the same adaptive
       WRITE.

State-reduction failure is used here as a falsification test of whether these
physical distinctions may be safely discarded.

It is not the physical mechanism itself.

The local response-rank audit further challenges whether the registered
adaptive response map admits a lower-dimensional smooth exact representation
in the tested neighborhood.

The bounded mechanistic statement is therefore:

    transient reconfiguration
        ->
    altered present physical state
        ->
    altered admissible subsequent transition
        ->
    failure of any tested reduction that removes the distinction responsible
    for that altered transition

This formulation is provisional until the targeted literature audit confirms
that this specific mechanistic combination and experimental framing are not
already directly established.

---

## 14. Article Construction Rule

The manuscript must be built in the following logical order:

    problem
        ->
    null hypothesis
        ->
    defeat criterion
        ->
    established mathematical framework
        ->
    registered physical system
        ->
    mechanism control
        ->
    retained-prestress test
        ->
    reduction-failure test
        ->
    stronger-reduction challenge
        ->
    bounded interpretation
        ->
    limitations
        ->
    future extensions

The architecture must follow the falsification problem.

The falsification problem must not be retrofitted to defend the architecture.

---

## 15. Final Control Invariant

Before any sentence is promoted into the manuscript, ask:

    Is this established theory?

    Is this directly demonstrated by a frozen Article III experiment?

    Is this only an interpretation of the demonstrated result?

    Is this an untested architectural extension?

These categories must never be silently merged.

The red line for Article III is:

    DO NOT TRY TO PROVE ROIF.

    TEST THE CONDITIONS UNDER WHICH ADDITIONAL STATE STRUCTURE IS OR IS NOT
    REQUIRED TO PRESERVE THE REGISTERED SUBSEQUENT TRANSITIONS.
