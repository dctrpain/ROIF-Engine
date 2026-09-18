# Article III — Hypothesis Core

## Purpose

This document defines the manuscript-facing scientific core of Article III.

It fixes:

1. the research question;
2. the null hypothesis;
3. the defeat criterion;
4. the registered test domain;
5. the contribution boundary.

It does not constitute the Abstract, Introduction, Results, or Discussion.

The physical object remains primary.

State reduction is used as a falsification instrument for determining which
distinctions in the present physical state are required to preserve the
registered subsequent transitions.

---

## 1. Physical Object

Article III studies a reconfiguring prestressed network.

The registered present state is written as:

    I_t = (P_t, C_t)

where:

    P_t = distributed prestress state

and:

    C_t = adaptive connection state.

For the present benchmark:

    I_t in R^13

with:

    P_t = (P_A, P_B, P_C)

and, for each of the two adaptive connections AB and BC:

    C = (s, c, r, f, b)

where:

    s = stiffness
    c = contractile capacity
    r = reflex gain
    f = fatigue
    b = remodeling bias

The 13-dimensional state is the registered state of the present benchmark.

It is not asserted to be a universal state representation or a globally
minimal representation.

Fixed model configuration is denoted by:

    Gamma

and contains the registered topology, reserves, bounds, adaptive
configuration, and redistribution configuration.

The transition is:

    I_(t+1) = T(I_t, e_t; Gamma)

where e_t is an admissible registered event.

The physical causal sequence under investigation is:

    perturbation
        ->
    redistribution through the current prestressed network
        ->
    adaptive reconfiguration
        ->
    changed present physical state
        ->
    changed response to a subsequent perturbation

---

## 2. Research Question

The primary research question is:

    When a prestressed network can reconfigure under admissible
    perturbations, can a state representation that is exact for the current
    propagation regime remain sufficient for subsequent transitions after
    reconfiguration?

Equivalently, for a projection:

    Z = Pi(I)

we ask whether the projected state remains closed under the registered
transition family.

The question is not whether reduction is possible in principle.

The question is:

    Which distinctions in the present physical state may be removed without
    changing the registered admissible subsequent transitions?

---

## 3. Null Hypothesis

The null hypothesis favors reduction.

For a candidate representation:

    Z = Pi(I)

the null hypothesis is:

    H0:

    there exists a reduced transition operator T_tilde such that

        Pi[T(I, e; Gamma)]
            =
        T_tilde(Pi(I), e; Gamma)

    over the registered state domain, event domain, and tested horizon.

Under H0, the information removed by Pi is unnecessary for predicting the
registered subsequent transitions.

Therefore, if H0 survives the registered falsification attempts, additional
internal state structure is not justified for that registered task.

This is the preferred outcome whenever the simpler representation is
sufficient.

---

## 4. Pairwise Transition-Sufficiency Criterion

For two registered states I_A and I_B, a necessary consequence of the null is:

    Pi(I_A) = Pi(I_B)

implies:

    Pi[T(I_A, e; Gamma)]
        =
    Pi[T(I_B, e; Gamma)]

for every registered admissible event e relevant to the tested claim.

For an event set E, define registered transition equivalence by:

    I_A ~_E I_B

if and only if:

    for every e in E,

        Pi[T(I_A, e; Gamma)]
            =
        Pi[T(I_B, e; Gamma)]

within the registered numerical tolerance.

The equivalence is explicitly conditioned on:

    the projection Pi;
    the event set E;
    the registered state domain;
    the observation used for comparison;
    the numerical tolerance;
    the tested transition horizon.

No stronger equivalence is implied.

---

## 5. Falsification Criterion for a Candidate Reduction

A candidate reduction Pi is falsified over the registered domain if a
registered collision can be constructed such that:

    Pi(I_A) = Pi(I_B)

while, for the same admissible event e:

    Pi[T(I_A, e; Gamma)]
        !=
    Pi[T(I_B, e; Gamma)]

beyond the registered numerical tolerance.

For a multi-step registered event sequence:

    e_(1:k) = (e_1, ..., e_k)

the same logic applies to the corresponding composed transition.

Therefore a representation may be sufficient at one transition horizon and
insufficient at another.

Failure of one candidate reduction does not establish that all lower-
dimensional reductions fail.

---

## 6. Defeat Criterion for the Stronger Claim

The stronger claim of Article III must be allowed to lose.

It is defeated for a registered task if a compact representation:

    Z = f(I)

can be specified such that its reduced dynamics preserve all registered
future distinctions required by that task over the registered event domain
and tested horizon.

Formally, if there exists an admissible reduced state Z and transition F such
that:

    Z_(t+1) = F(Z_t, e_t)

and the registered future observations are preserved for the full registered
test domain, then the additional state distinctions removed by f are not
required for that task.

The compact representation is allowed to be:

    nonlinear;
    physically transformed;
    latent;
    learned;
    a sufficient statistic;
    a predictive state;

provided that it is specified independently of the particular test outcome
being used to defend it and is evaluated under the same registered event
domain.

The stronger claim must not be rescued after failure by adding hidden
variables post hoc without a new explicitly registered test.

The manuscript therefore does not assume that the original physical
coordinates are the unique sufficient coordinates.

---

## 7. Registered Event-Domain Dependence

Transition sufficiency is event-domain dependent.

A representation may satisfy:

    H0 over E_direct

while failing over:

    E_direct union E_write

where E_write contains events capable of changing the adaptive state.

This distinction is central to the registered Z7 experiment.

The relevant question is therefore not:

    Is Z sufficient?

but:

    Is Z sufficient for this registered event domain and transition horizon?

The manuscript must state the event domain whenever a sufficiency claim is
made.

---

## 8. Registered Physical Reduction Challenge

The principal explicit reduction challenge is:

    Z7 =
    (
        P_A,
        P_B,
        P_C,
        g_AB,
        c_AB,
        g_BC,
        c_BC
    )

with:

    g = s * c * r * (1 - f)

for the registered adaptive connection state:

    C = (s, c, r, f, b)

Z7 preserves the tested current mechanical interface for the constructed
collision.

The registered direct mechanical probe does not distinguish the two source
states.

Therefore the reduction is not rejected by that probe.

The falsification challenge occurs only when the admissible event domain is
expanded to include adaptive WRITE followed by subsequent mechanical READ.

The relevant sequence is:

    I_A != I_B
        ->
    Z7_A = Z7_B
        ->
    same direct READ
        ->
    same registered mechanical response
        ->
    same adaptive WRITE
        ->
    Z7_A' != Z7_B'
        ->
    same future READ
        ->
    different registered mechanical response

This tests closure under reconfiguration rather than merely information loss
under projection.

The frozen registered decision is:

    Z7_NOT_TRANSITION_SUFFICIENT_OVER_REGISTERED_DOMAIN

---

## 9. Retained-Prestress Challenge

A second physical challenge asks whether the adaptive connection state alone
is sufficient when the distributed prestress state is discarded.

The tested reduction is conceptually:

    (P, C) -> C

The registered experiment constructs two perturbation sequences for which:

    C_AB = C_BA

while:

    P_AB != P_BA

The subsequent locked READ preserves the registered distinction in the
prestress state while the adaptive connection states remain equal.

Therefore, for the registered READ:

    (P, C) -> C

is not transition-sufficient.

The supported interpretation is:

    effects of preceding events may remain physically encoded in the present
    distributed state even when the registered adaptive connection state is
    identical.

This does not imply reconstruction of the complete historical trajectory.

---

## 10. Mechanism Control

The mechanism-decomposition experiment separates three regimes:

    additive reference;
    fixed-network production redistribution;
    adaptive reconfiguration.

For the registered AB/BA sequence comparison:

    D_additive = 0

    D_fixed = 2.77555756e-17

    D_reconfiguring = 0.011814192037381752

with registered tolerance:

    1e-12

Thus the registered sequence distinction appears in the tested
reconfiguration regime while remaining absent within tolerance in the two
registered controls.

This is a mechanism-specific computational result.

It is not a general theorem that fixed networks are order-independent or that
reconfiguration is necessary for path dependence in dynamical systems.

---

## 11. Stronger Compression Challenge

The local response-rank audit challenges whether the registered
five-dimensional adaptive connection state:

    C = (s, c, r, f, b)

admits a smooth exact local reduction below five dimensions for the
registered future-response map.

The numerical response Jacobian has rank:

    5

at all pre-specified finite-difference step sizes:

    h = 1e-5
    h = 1e-6
    h = 1e-7

Because the registered future-response map has full differential rank with
respect to the five adaptive-state coordinates at the tested interior point,
the numerical result excludes, under the registered smooth exact-reduction
assumptions, a locally smooth exact factorization of that response map through
a state space of dimension below five in the tested neighborhood.

This is a differential-rank statement about the registered finite
future-response map. Its interpretation is consistent with established local
observability and minimal-realization ideas, but the numerical audit is not
presented as an application or proof of the full Hermann-Krener observability
rank condition.

It does not establish:

    global minimality;
    global observability;
    impossibility of non-smooth reductions;
    uniqueness of the physical coordinates;
    minimality of the complete 13-dimensional registered state.

---

## 12. What Would Count as Defeat

The manuscript must explicitly recognize the following outcomes as defeats
of stronger interpretations.

The stronger interpretation is defeated if:

    a lower-dimensional representation preserves the complete registered
    transition family relevant to the claim;

or:

    a hidden distinction removed by the candidate representation never
    changes any registered future transition over the tested event domain;

or:

    the observed divergence disappears under the registered negative or
    mechanism controls;

or:

    the claimed reduction failure depends only on a numerical artifact,
    tolerance choice, non-determinism, or boundary violation;

or:

    the local rank result is not stable under the pre-specified numerical
    perturbation scales.

A simpler sufficient representation is not a failure of the experiment.

It is a scientifically valid result against the necessity of the additional
state structure.

---

## 13. What the Frozen Suite Actually Establishes

Within the registered computational system, the frozen suite supports the
following bounded statements.

First:

    the tested reconfiguration mechanism introduces registered AB/BA order
    distinguishability absent within tolerance in the additive and fixed-
    network controls.

Second:

    sequence-dependent distributed prestress can remain transition-relevant
    even when the registered terminal adaptive connection state is identical.

Third:

    Z7 is exact for the constructed current direct mechanical probe but is
    not closed under the larger registered event domain containing adaptive
    WRITE followed by future READ.

Fourth:

    the registered five-dimensional adaptive response map has stable local
    numerical rank five under the pre-specified audit.

These results are computational properties of the registered model and
experiments.

---

## 14. Contribution Boundary

The contribution is not a new general theory of state reduction.

The contribution is not the discovery of:

    bisimulation;
    lumpability;
    observability;
    predictive-state representations;
    minimal realization;
    path dependence;
    hysteresis;
    prestress-dependent response;
    adaptive networks.

The bounded computational contribution is:

    a falsifiable mechanistic demonstration of how transient reconfiguration
    in a prestressed network can expose present-state distinctions that are
    silent under a current mechanical probe but become relevant to later
    registered transitions.

Two concrete routes are demonstrated in the frozen suite:

    retained distributed prestress;

and:

    latent adaptive decomposition exposed by subsequent reconfiguration.

The reduction tests determine whether those physical distinctions may be
discarded for the registered future-transition problem.

State-reduction failure is therefore evidence about the physical mechanism.

It is not itself the mechanism.

---

## 15. Scope Boundary

The frozen suite does not establish:

    biological validity;
    clinical validity;
    universality across complex systems;
    global state minimality;
    uniqueness of the registered coordinates;
    impossibility of all alternative reductions;
    reconstruction of the complete historical trajectory;
    a general diagnostic active-probing algorithm.

The current conclusions apply to:

    the registered computational model;
    the registered state domain;
    the registered event domain;
    the registered observation map;
    the registered numerical tolerance;
    the tested transition horizons.

Any extension beyond these boundaries requires a separate argument or test.

---

## 16. Manuscript Invariant

The manuscript must preserve the following order of explanation:

    physical mechanism
        ->
    reduction question
        ->
    null hypothesis
        ->
    defeat criterion
        ->
    registered falsification experiments
        ->
    bounded interpretation

It must not invert this into:

    abstract reduction theory
        ->
    generic complex-systems claim
        ->
    ROIF as confirmation.

The central control statement is:

    IF A SIMPLER REPRESENTATION PRESERVES THE REGISTERED SUBSEQUENT
    TRANSITIONS, THE ADDITIONAL STATE STRUCTURE IS NOT REQUIRED FOR THAT
    REGISTERED TASK.

The physical counterpart is:

    A PRESENT-STATE DISTINCTION SHOULD BE RETAINED ONLY WHEN REMOVING IT
    CHANGES AN ADMISSIBLE SUBSEQUENT TRANSITION WITHIN THE REGISTERED DOMAIN.