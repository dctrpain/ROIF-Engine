# ARTICLE III — MASTER PROTOCOL

Status: CONTROL DOCUMENT
Purpose: preserve the physical basis, falsification logic, novelty boundary,
and reviewer reproducibility requirements of Article III.

This document is NOT the manuscript.
Any later change that contradicts this protocol must be explicit and justified.

---

## 1. PRIMARY SCIENTIFIC OBJECT

The primary object is not "memory", "history", or ROIF as a brand.

The object is a dynamical process X(t) that extends outside the temporal
slice directly accessible to an observer.

                         Sigma(t0)
                            |
        X(t < t0) ----------+----------> X(t > t0)

At the current slice, only partial observations may be available:

    Y_i(t0) = Pi_i[X(t0)],    i = 1,...,m.

The past process need not be uniquely reconstructible.

The central problem is to determine which distinctions present in, or hidden
behind, the current slice must be preserved because eliminating them changes
admissible subsequent transitions.

This temporal-slice formulation is a conceptual problem statement.
It is NOT claimed as a new general theory.

---

## 2. PHYSICAL INVARIANT

No mathematical abstraction is allowed to replace the physical object that
it abstracts without explicitly stating what information has been removed.

For the minimal registered computational realization:

    X_t = (P_t, C_t)

where

    P_t = distributed prestress state
    C_t = internal adaptive connection state.

The registered recursive transition has the form

    C_(t+1) = U(C_t, e_t)

    G_(t+1) = G(C_(t+1))

    P_(t+1) = R(P_t, e_t; G_(t+1))

and therefore

    (P_t, C_t) --e_t--> (P_(t+1), C_(t+1)).

P and C are NOT asserted to exhaust ROIF, biological systems,
or general complex systems.

They form the minimal physical state used by the present falsification
experiments.

---

## 3. ORIGINAL ALGORITHMIC FOUNDATION MUST NOT BE LOST

The broader physical architecture motivating the engine is:

    external/internal influences
        ->
    material state
        ->
    prestressed configuration
        ->
    redistribution
        ->
    adaptation/reconfiguration
        ->
    changed subsequent response.

In more general notation:

    Theta_t
        ->
    M_t
        ->
    (P_t, G_t)
        ->
    R_t
        ->
    M_(t+1)
        ->
    (P_(t+1), G_(t+1)).

Possible influence fields include temperature, hydration/moisture,
composition, mechanical loading, damage, recovery, aging/time,
and biological or chemical influences.

Article III does NOT claim that all of these fields are implemented or
validated by the present experiments.

They belong to the broader physical architecture from which the minimal
testable problem was extracted.

---

## 4. FUNCTIONAL PLANES ARE OBSERVATIONAL DOMAINS, NOT STATE DIMENSIONS

The medical implementation contains functional domains n_1,...,n_13.

These must NOT be identified with the numerical dimension of X.

In the general abstraction they are treated as possible observational
projections:

    Y_i = Pi_i(X).

The number 13 belongs to a particular medical architecture.
It is NOT claimed to be a universal dimension of complex systems.

Therefore:

    functional domains
        != influence fields
        != material variables
        != dynamic network state.

The original idea of probing one process through multiple planes is
preserved without requiring Article III to establish the universality
of the thirteen medical planes.

---

## 5. CENTRAL SCIENTIFIC QUESTION

Article III asks:

Which distinctions of state, potentially hidden at the currently observed
temporal slice, must be preserved because eliminating them changes subsequent
admissible transitions of a reconfiguring prestressed system?

Let

    Pi(X_A) = Pi(X_B).

The operational question is whether there exists a registered event sequence
E in event class E such that

    O[T^k(X_A, E)] != O[T^k(X_B, E)].

The article studies this question for a concrete executable transition
operator.

It does NOT claim a new general theory of state sufficiency.

---

## 6. DEFEAT CONDITION

The strong hypothesis must be allowed to lose.

If a simpler representation

    Z = f(X)

exists such that equality

    Z_A = Z_B

preserves all registered future outputs over the specified event class,
horizon, output map, and tolerance, then the additional state structure is
unnecessary for that registered task.

Such a result is accepted as defeat of the stronger representation claim.

Post-result rescue by introducing unregistered hidden variables,
changing the event class, changing the tolerance, or changing the output
criterion is prohibited.

---

## 7. EXPERIMENTAL MECHANISM A:
## RETAINED DISTRIBUTED PRESTRESS

Two registered event orders AB and BA produce identical terminal adaptive
connection states:

    D_C(AB, BA) = 0

while their distributed prestress states differ:

    D_P(AB, BA) = 0.011814192037381752.

After an identical registered READ event:

    D_C(ABR, BAR) = 0

and

    D_P(ABR, BAR) = 0.011814192037381738.

Therefore, for the registered event regime,

    (P, C) -> C

is not transition-sufficient.

Allowed interpretation:

Information about the preceding process can remain physically represented
in distributed prestress even when the registered terminal adaptive
connection state is identical.

Forbidden interpretation:

Prestress is universally necessary in every dynamical or mechanical system.

---

## 8. EXPERIMENTAL MECHANISM B:
## LATENT RECONFIGURATION STATE

For one adaptive connection

    C = (s, c, r, f, b)

and the registered current mechanical interface is

    Q(C) = (g, c)

with

    g = s*c*r*(1-f).

Construct two distinct internal states

    C_A != C_B

such that

    Q(C_A) = Q(C_B).

The registered direct mechanical probe produces identical physical
continuation:

    D_P = 0.

Thus the reduction is exact for the tested current-propagation regime.

After the same registered WRITE event,

    C'_A = U(C_A, W)
    C'_B = U(C_B, W)

the mechanical interfaces diverge:

    D(Q(C'_A), Q(C'_B))
        = 0.033213859765625076.

After the same subsequent READ event:

    D_P = 0.0077844983825683944.

Therefore:

    Q(C_A) = Q(C_B)

does NOT imply

    Q(U(C_A,W)) = Q(U(C_B,W))

for the registered reconfiguration event.

Mechanistic sequence:

    exact current interface equivalence
        ->
    identical reconfiguration
        ->
    interface divergence
        ->
    future physical divergence.

---

## 9. MECHANISM CONTROL

Registered event-order distances:

    D_additive = 0

    D_fixed =
        2.7755575615628914e-17

    D_reconfiguring =
        0.011814192037381752.

Allowed claim:

In the registered computational construction, measured order dependence is
absent within tolerance in the additive and fixed-network controls and
appears when intermediate reconfiguration is enabled.

Forbidden claim:

Reconfiguration is the universal cause of path dependence.

---

## 10. LOCAL RESPONSE-RANK AUDIT

For the registered adaptive connection state

    C = (s,c,r,f,b)

the local future-response map Phi(C) was evaluated using the registered
probe set.

At the registered interior state, the numerical Jacobian rank is

    rank J_Phi = 5

for finite-difference steps

    h = 1e-5, 1e-6, 1e-7.

This is supporting falsification evidence.

Allowed claim:

The registered numerical response map provides a local lower-bound
certificate excluding an exact smooth reduction below five dimensions
near the tested state under the registered rank criterion.

Forbidden claims:

    - the full ROIF state is globally minimal;
    - every exact representation requires five variables;
    - non-smooth reductions are excluded.

---

## 11. ESTABLISHED THEORY IS NOT OUR DISCOVERY

Article III must explicitly distinguish its results from established work
in, at minimum:

    state sufficiency
    observability
    nonlinear observability
    bisimulation
    lumpability
    predictive-state representations
    internal-variable mechanics
    hysteresis
    path dependence
    adaptive/co-evolving networks
    prestress-dependent mechanics
    mechanical memory
    model reduction
    active experiment design.

No established concept may be renamed and presented as a ROIF discovery.

Targeted literature audit is required before final novelty wording.

---

## 12. PROVISIONAL CONTRIBUTION BOUNDARY

The present computational contribution is provisionally centered on two
distinct mechanisms of reduction failure in the same registered recursive
prestressed operator:

    A. retained distributed physical state;

    B. latent internal state exposed by an admissible reconfiguration.

These mechanisms demonstrate two different locations in which distinctions
invisible to a particular current reduction may remain relevant to later
transitions.

The novelty of this exact combination and mechanistic construction must be
established by targeted literature audit before submission.

---

## 13. THE ENGINE IS THE COMPUTATIONAL EXPERIMENTAL APPARATUS

ROIF does not need to be the theoretical subject or branding of Article III.

The engine is instead the executable computational apparatus on which the
registered hypotheses can be independently tested.

Required chain:

    manuscript claim
        <->
    preregistration
        <->
    executable benchmark
        <->
    production modules exercised
        <->
    machine-readable result
        <->
    regression test.

The article must not ask reviewers merely to trust reported numerical
results when those results can be reproduced directly.

---

## 14. REVIEWER REPRODUCIBILITY REQUIREMENT

Before submission there must be a reviewer-facing reproduction path.

Target interface:

    python -m experiments.run_manuscript_suite

The suite should reproduce all central registered experiments and report
their status without modifying production state or experimental conditions.

Target summary:

    MANUSCRIPT REPRODUCTION SUITE

    Mechanism decomposition ............. PASS
    Retained prestress .................. PASS
    Z7 reduction falsification .......... PASS
    Local response-rank audit ........... PASS

    Registered falsification suite reproduced.

    Biological validity tested: NO
    Clinical validity tested: NO
    Global minimality established: NO

The exact runner must be built only after inspecting and reusing the
existing frozen benchmark modules.

---

## 15. PRODUCTION / EXPERIMENT BOUNDARY

The manuscript must distinguish:

    experimental harness

from

    production transition operator.

Experimental code may:

    - construct initial states;
    - construct collision pairs;
    - register events;
    - calculate comparison metrics;
    - write result files.

It must be explicit which production modules execute the transition being
tested.

No central result may be described as a production-engine property if the
relevant behavior exists only inside experimental scaffolding.

This requirement directly addresses the risk of circular synthetic
validation.

---

## 16. HISTORICAL INITIATOR VS CURRENT MAINTAINING STATE

A historical initiating process need not remain active at the current
temporal slice.

A preceding process may modify:

    P, C, M, ...

such that subsequent dynamics are governed by the altered current state.

Therefore:

    historical initiating process
        !=
    current transition-maintaining state.

Article III does NOT claim unique reconstruction of a historical D0.

This distinction is essential for connecting the computational formulation
to the causal problem that motivated the original architecture.

---

## 17. ACTIVE PROBING

If several states collide under the current observation,

    Pi(X_A) = Pi(X_B),

a future probe may distinguish them.

Conceptually:

    e* = argmax_e Distinguishability(
            O[T(X_A,e)],
            O[T(X_B,e)]
         ).

This connects the present work to active probing and experiment design.

Unless automatic probe selection is directly tested in Article III,
Active Probe Engine remains an architectural extension / future-work
direction rather than a demonstrated central result.

---

## 18. FOUR QUESTIONS THAT MUST SURVIVE EVERY ABSTRACTION

Before accepting any new mathematical abstraction in Article III, answer:

1. What physical object does X represent?

2. What does Pi actually observe or remove?

3. What physical/computational process does T represent?

4. What part of the original algorithm is lost by this abstraction?

If question 4 cannot be answered precisely, manuscript development stops
until the abstraction is understood.

No mathematical convenience is allowed to silently replace the physical
algorithm.

---

## 19. ARTICLE III READINESS CONDITIONS

Full manuscript writing begins only when all four conditions are satisfied:

    [ ] Physical invariant fixed
    [ ] Novelty boundary audited
    [ ] Central falsification suite frozen
    [ ] Reviewer reproduction path operational

Only then proceed to:

    Title
        ->
    Abstract
        ->
    Introduction
        ->
    Methods
        ->
    Results
        ->
    Discussion
        ->
    Limitations
        ->
    Conclusion.

---

## 20. RED LINE

DO NOT TRY TO PROVE ROIF.

Try to eliminate the need for its additional structure.

If a simpler representation survives the registered falsification domain,
accept it.

If additional structure survives only because the experiment was designed
to favor it, redesign the experiment.

If an abstraction makes the manuscript cleaner but removes the physical
basis of the algorithm, reject the abstraction.

If a central computational claim cannot be independently reproduced on the
engine, it is not ready for the manuscript.

---

END OF MASTER PROTOCOL
