
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

GENERATOR_VERSION = "roif_entropy_assets_v2"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"


class PublicationAssetError(RuntimeError):
    pass


def load_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise PublicationAssetError(f"Missing benchmark file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(mapping: dict[str, Any], *path: str) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise PublicationAssetError("Missing field: " + ".".join(path))
        cur = cur[key]
    return cur


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise PublicationAssetError(f"{label} must be finite")
    return number


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "pdf": str(pdf.relative_to(ROOT)),
        "png": str(png.relative_to(ROOT)),
    }


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        text = text.replace(old, new)
    return text


def write_latex_table(
    path: Path,
    headers: list[str],
    rows: list[list[Any]],
    *,
    caption: str,
    label: str,
    column_spec: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if column_spec is None:
        column_spec = "l" * len(headers)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(x) for x in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def figure_global_redistribution(physical: dict[str, Any]) -> dict[str, str]:
    section = require(physical, "global_redistribution")
    deltas = require(section, "all_node_deltas")
    nodes = sorted(deltas)
    values = [finite(deltas[n], f"delta.{n}") for n in nodes]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(nodes, values)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Prestress node")
    ax.set_ylabel("Prestress change")
    ax.set_title("Local perturbation produces downstream prestress redistribution")
    direct = ", ".join(require(section, "directly_perturbed_nodes"))
    ax.text(
        0.5, -0.18,
        f"Directly perturbed node(s): {direct}",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=8,
    )
    return save_figure(fig, "figure_2_global_redistribution")


def figure_path_dependence(physical: dict[str, Any]) -> dict[str, str]:
    section = require(physical, "order_dependence")
    ab = dict(require(section, "ab_signature")[3])
    ba = dict(require(section, "ba_signature")[3])
    nodes = sorted(set(ab) | set(ba))
    ab_values = [finite(ab[n], f"AB.{n}") for n in nodes]
    ba_values = [finite(ba[n], f"BA.{n}") for n in nodes]

    x = list(range(len(nodes)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar([i - width / 2 for i in x], ab_values, width=width, label="A → B")
    ax.bar([i + width / 2 for i in x], ba_values, width=width, label="B → A")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x, nodes)
    ax.set_xlabel("Prestress node")
    ax.set_ylabel("Final prestress")
    ax.set_title("Event order changes the final prestress state")
    ax.legend()
    distance = finite(require(section, "final_state_distance"), "final_state_distance")
    ax.text(
        0.5, -0.18,
        f"Final-state distance = {distance:.6g}",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=8,
    )
    return save_figure(fig, "figure_3_path_dependence")


def figure_repeated_event(physical: dict[str, Any]) -> dict[str, str]:
    section = require(physical, "repeated_event_state_dependence")
    first = require(section, "first_response")
    second = require(section, "second_response")
    metrics = [
        ("connection_change_norm", "Connection change"),
        ("prestress_change_norm", "Prestress change"),
        ("trace_magnitude", "Trace magnitude"),
    ]
    labels = [label for _, label in metrics]
    a = [finite(first[k], f"first.{k}") for k, _ in metrics]
    b = [finite(second[k], f"second.{k}") for k, _ in metrics]

    x = list(range(len(metrics)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar([i - width / 2 for i in x], a, width=width, label="First exposure")
    ax.bar([i + width / 2 for i in x], b, width=width, label="Second identical exposure")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Response norm / magnitude")
    ax.set_title("The same event produces a different response after history")
    ax.legend()
    distance = finite(require(section, "response_distance"), "response_distance")
    ax.text(
        0.5, -0.18,
        f"Response distance = {distance:.6g}",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=8,
    )
    return save_figure(fig, "figure_4_repeated_event")


def _find_separability_models(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Locate the Q5 model block without depending on one fixed JSON nesting path.

    Accepted payloads include:
    - {"separability": {...models...}}
    - {...models...}
    - any nested object containing all four required model names
    """
    required = {
        "separable_additive",
        "nonlinear_but_separable",
        "multiplicative_interaction",
        "state_modulated_interaction",
    }

    if required.issubset(payload.keys()):
        return payload

    direct = payload.get("separability")
    if isinstance(direct, dict) and required.issubset(direct.keys()):
        return direct

    stack = [payload]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if required.issubset(current.keys()):
            return current
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)

    raise PublicationAssetError(
        "Could not locate Q5 separability model block. "
        "Expected model keys: " + ", ".join(sorted(required))
    )



def _looks_like_q6_gain_block(candidate: dict[str, Any]) -> bool:
    if not candidate:
        return False

    required_metric_keys = {
        "all_bounded_under_limit",
        "max_peak_amplification_ratio",
        "positive_max_lambda_count",
    }

    numeric_like_key_count = 0
    valid_metric_block_count = 0

    for key, value in candidate.items():
        try:
            float(key)
            numeric_like_key_count += 1
        except (TypeError, ValueError):
            continue

        if (
            isinstance(value, dict)
            and required_metric_keys.issubset(value.keys())
        ):
            valid_metric_block_count += 1

    return (
        numeric_like_key_count > 0
        and numeric_like_key_count == valid_metric_block_count
    )


def _find_q6_by_feedback_gain(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Locate the Q6 feedback-gain summary block without assuming one fixed
    JSON nesting path.

    Accepted forms include:
    - {"by_feedback_gain": {...}}
    - a nested object containing "by_feedback_gain"
    - the gain dictionary itself, e.g. {"0.0": {...}, "1.0": {...}}
    """
    direct = payload.get("by_feedback_gain")
    if isinstance(direct, dict) and _looks_like_q6_gain_block(direct):
        return direct

    if _looks_like_q6_gain_block(payload):
        return payload

    stack = [payload]
    while stack:
        current = stack.pop()

        if not isinstance(current, dict):
            continue

        direct = current.get("by_feedback_gain")
        if isinstance(direct, dict) and _looks_like_q6_gain_block(direct):
            return direct

        if _looks_like_q6_gain_block(current):
            return current

        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)

    raise PublicationAssetError(
        "Could not locate Q6 feedback-gain summary block. "
        "Expected numeric gain keys containing "
        "all_bounded_under_limit, max_peak_amplification_ratio, "
        "and positive_max_lambda_count."
    )


def figure_separability(separability: dict[str, Any]) -> dict[str, str]:
    models = _find_separability_models(separability)
    order = [
        "separable_additive",
        "nonlinear_but_separable",
        "multiplicative_interaction",
        "state_modulated_interaction",
    ]
    labels_map = {
        "separable_additive": "Additive\nseparable",
        "nonlinear_but_separable": "Nonlinear\nseparable",
        "multiplicative_interaction": "Multiplicative\ninteraction",
        "state_modulated_interaction": "State-modulated\ninteraction",
    }
    for name in order:
        if name not in models:
            raise PublicationAssetError(f"Missing separability model: {name}")

    values = [
        finite(
            require(models[name], "mean_abs_interaction_residual"),
            f"{name}.mean_abs_interaction_residual",
        )
        for name in order
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar([labels_map[n] for n in order], values)
    ax.set_ylabel("Mean absolute interaction residual")
    ax.set_title("Separability control distinguishes separable and interacting models")
    return save_figure(fig, "figure_5_separability")


def figure_q6(q6: dict[str, Any]) -> dict[str, str]:
    by_gain = _find_q6_by_feedback_gain(q6)
    keys = sorted(by_gain, key=lambda x: float(x))
    gains = [float(k) for k in keys]
    peak = [
        finite(
            require(by_gain[k], "max_peak_amplification_ratio"),
            f"{k}.max_peak_amplification_ratio",
        )
        for k in keys
    ]
    positive = [int(require(by_gain[k], "positive_max_lambda_count")) for k in keys]

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.plot(gains, peak, marker="o")
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xlabel("Feedback gain")
    ax.set_ylabel("Maximum peak amplification ratio")
    ax.set_title("Q6 perturbations remain bounded in the tested regime")
    ax.text(
        0.5, -0.18,
        "Positive max Lyapunov-like counts: "
        + ", ".join(f"{g:g}:{c}" for g, c in zip(gains, positive)),
        transform=ax.transAxes,
        ha="center", va="top", fontsize=8,
    )
    return save_figure(fig, "figure_6_q6_feedback_gain")




def figure_temporal_reconstruction(
    temporal: dict[str, Any],
) -> dict[str, str]:
    """
    Figure 7.

    Compare reconstruction error for:
    - full ROIF,
    - linear interpolation,
    - history-aware fixed coupling,
    - memory-free replay.

    Exact full-ROIF replay is an internal consistency result because the
    synthetic truth and the full replay use the same deterministic transition
    model.
    """
    section = require(temporal, "temporal_reconstruction")
    methods = require(section, "methods")

    order = [
        "full_roif",
        "linear_interpolation",
        "history_aware_fixed_coupling",
        "memory_free",
    ]

    labels = [
        "Full ROIF",
        "Linear\ninterpolation",
        "History-aware\nfixed coupling",
        "Memory-free",
    ]

    hidden = [
        finite(
            require(methods[name], "mean_hidden_slice_rmse"),
            f"{name}.mean_hidden_slice_rmse",
        )
        for name in order
    ]

    future = [
        finite(
            require(methods[name], "future_slice_rmse"),
            f"{name}.future_slice_rmse",
        )
        for name in order
    ]

    x = list(range(len(order)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.bar(
        [i - width / 2 for i in x],
        hidden,
        width=width,
        label="Withheld intermediate slices",
    )

    ax.bar(
        [i + width / 2 for i in x],
        future,
        width=width,
        label="Withheld future slice",
    )

    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE in normalized system-image vector")
    ax.set_title(
        "Temporal image reconstruction under controlled synthetic dynamics"
    )
    ax.legend()

    ax.text(
        0.5,
        -0.23,
        (
            "Full replay shares the deterministic transition model used to "
            "generate synthetic truth; zero error is an internal-consistency "
            "result, not external forecasting validation."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    return save_figure(
        fig,
        "figure_7_temporal_image_reconstruction",
    )


def figure_operator_evolution(
    temporal: dict[str, Any],
) -> dict[str, str]:
    """
    Figure 8.

    Compare A->B->C versus B->A->C.

    The same C event is applied after two different histories.
    """
    section = require(temporal, "operator_evolution")

    metrics = [
        ("pre_c_state_distance", "Pre-C\nstate"),
        ("post_c_state_distance", "Post-C\nstate"),
        ("c_response_distance", "C-response"),
        ("trajectory_distance", "Whole\ntrajectory"),
    ]

    labels = [label for _, label in metrics]

    values = [
        finite(
            require(section, key),
            f"operator_evolution.{key}",
        )
        for key, _ in metrics
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.9))

    ax.bar(labels, values)

    ax.set_ylabel("Euclidean distance")
    ax.set_title(
        "Event history changes the response to the same subsequent event C"
    )

    ax.text(
        0.5,
        -0.20,
        (
            "Comparison: A -> B -> C versus B -> A -> C; "
            "the same event C is used in both histories."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    return save_figure(
        fig,
        "figure_8_operator_evolution",
    )


def table_temporal_reconstruction(
    temporal: dict[str, Any],
) -> dict[str, str]:
    section = require(temporal, "temporal_reconstruction")
    methods = require(section, "methods")

    order = [
        "full_roif",
        "linear_interpolation",
        "history_aware_fixed_coupling",
        "memory_free",
    ]

    rows = []

    for name in order:
        block = require(methods, name)

        rows.append(
            [
                name,
                f"{finite(require(block, 'mean_hidden_slice_rmse'), name + '.hidden'):.9g}",
                f"{finite(require(block, 'future_slice_rmse'), name + '.future'):.9g}",
                f"{finite(require(block, 'mean_all_target_rmse'), name + '.all'):.9g}",
            ]
        )

    headers = [
        "Method",
        "Hidden-slice RMSE",
        "Future-slice RMSE",
        "All-target RMSE",
    ]

    csv_path = TABLES / "table_temporal_reconstruction.csv"
    tex_path = TABLES / "table_temporal_reconstruction.tex"

    write_csv(csv_path, headers, rows)

    write_latex_table(
        tex_path,
        headers,
        rows,
        caption=(
            "Temporal image reconstruction under controlled synthetic dynamics."
        ),
        label="tab:temporal-reconstruction",
        column_spec=r"p{0.40\textwidth}p{0.14\textwidth}p{0.14\textwidth}p{0.14\textwidth}",
    )

    return {
        "csv": str(csv_path.relative_to(ROOT)),
        "tex": str(tex_path.relative_to(ROOT)),
    }


def table_operator_evolution(
    temporal: dict[str, Any],
) -> dict[str, str]:
    section = require(temporal, "operator_evolution")

    rows = [
        [
            "Pre-C state distance",
            f"{finite(require(section, 'pre_c_state_distance'), 'pre_c'):.9g}",
        ],
        [
            "Post-C state distance",
            f"{finite(require(section, 'post_c_state_distance'), 'post_c'):.9g}",
        ],
        [
            "C-response distance",
            f"{finite(require(section, 'c_response_distance'), 'c_response'):.9g}",
        ],
        [
            "Whole-trajectory distance",
            f"{finite(require(section, 'trajectory_distance'), 'trajectory'):.9g}",
        ],
    ]

    headers = [
        "Metric",
        "Value",
    ]

    csv_path = TABLES / "table_operator_evolution.csv"
    tex_path = TABLES / "table_operator_evolution.tex"

    write_csv(csv_path, headers, rows)

    write_latex_table(
        tex_path,
        headers,
        rows,
        caption=(
            "History-conditioned response: "
            "A to B to C versus B to A to C."
        ),
        label="tab:operator-evolution",
    )

    return {
        "csv": str(csv_path.relative_to(ROOT)),
        "tex": str(tex_path.relative_to(ROOT)),
    }



def table_physical_history(physical: dict[str, Any]) -> dict[str, str]:
    g = require(physical, "global_redistribution")
    o = require(physical, "order_dependence")
    r = require(physical, "repeated_event_state_dependence")
    rows = [
        [
            "Local-to-global redistribution",
            "Prestress change norm",
            f"{float(g['prestress_change_norm']):.9g}",
            str(bool(g["global_redistribution_detected"])),
        ],
        [
            "Order dependence: A→B vs B→A",
            "Final-state distance",
            f"{float(o['final_state_distance']):.9g}",
            str(bool(o["path_dependent"])),
        ],
        [
            "Repeated identical event",
            "Response distance",
            f"{float(r['response_distance']):.9g}",
            str(bool(r["same_event_different_response"])),
        ],
    ]
    headers = ["Experiment", "Metric", "Value", "Detected"]
    csv_path = TABLES / "table_physical_history.csv"
    tex_path = TABLES / "table_physical_history.tex"
    write_csv(csv_path, headers, rows)
    write_latex_table(
        tex_path, headers, rows,
        caption="Physical-history integration benchmark.",
        label="tab:physical-history",
    )
    return {
        "csv": str(csv_path.relative_to(ROOT)),
        "tex": str(tex_path.relative_to(ROOT)),
    }


def table_claim_boundaries() -> dict[str, str]:
    rows = [
        ["Path dependence in the computational model", "Clinical causality"],
        ["State-dependent repeated-event response", "Universal biological law"],
        ["Prestress redistribution in the explicit model network", "Validation in living tissue"],
        ["Non-separable interaction in selected controls", "Universal multiplicativity"],
        ["Bounded/decaying Q6 perturbations in tested regimes", "Deterministic chaos or butterfly effect"],
        ["Software correctness under regression tests", "Empirical validation of model assumptions"],
        [
            "Exact deterministic replay of withheld synthetic slices",
            "External forecasting accuracy",
        ],
        [
            "History-conditioned response to the same subsequent event",
            "Universal causal law",
        ],
    ]
    headers = ["ROIF demonstrates", "ROIF does not claim"]
    csv_path = TABLES / "table_claim_boundaries.csv"
    tex_path = TABLES / "table_claim_boundaries.tex"
    write_csv(csv_path, headers, rows)
    write_latex_table(
        tex_path, headers, rows,
        caption="Claim boundaries for the present computational study.",
        label="tab:claim-boundaries",
        column_spec=r"p{0.44\textwidth}p{0.44\textwidth}",
    )
    return {
        "csv": str(csv_path.relative_to(ROOT)),
        "tex": str(tex_path.relative_to(ROOT)),
    }


def table_benchmark_summary(
    physical: dict[str, Any],
    separability: dict[str, Any],
    q6: dict[str, Any],
    temporal: dict[str, Any],
) -> dict[str, str]:
    models = _find_separability_models(separability)
    for name in (
        "separable_additive",
        "nonlinear_but_separable",
        "multiplicative_interaction",
        "state_modulated_interaction",
    ):
        if name not in models:
            raise PublicationAssetError(f"Missing separability model: {name}")

    q6_by_gain = _find_q6_by_feedback_gain(q6)
    all_bounded = all(
        bool(block["all_bounded_under_limit"])
        for block in q6_by_gain.values()
    )

    rows = [
        [
            "Q4",
            "Does abrupt threshold behavior uniquely establish multiplicativity?",
            "No",
            "Negative control",
        ],
        [
            "Q5",
            "Can separability distinguish independent from interacting models?",
            "Yes",
            "Non-separability",
        ],
        [
            "Q6",
            "Do tested perturbations show sensitive/unbounded growth?",
            "No; all tested gains bounded" if all_bounded else "Mixed",
            "Chaos not demonstrated",
        ],
        [
            "Physical 1",
            "Can a local prestress perturbation propagate downstream?",
            str(bool(require(physical, "global_redistribution", "global_redistribution_detected"))),
            "Distributed prestress response",
        ],
        [
            "Physical 2",
            "Does event order change the final system state?",
            str(bool(require(physical, "order_dependence", "path_dependent"))),
            "Path dependence / non-commutativity",
        ],
        [
            "Physical 3",
            "Does the same event give the same response after state change?",
            "No" if bool(require(
                physical,
                "repeated_event_state_dependence",
                "same_event_different_response",
            )) else "Yes",
            "State-dependent response",
        ],
        [
            "Q7",
            (
                "Can withheld temporal slices be reconstructed under "
                "known model dynamics?"
            ),
            "Yes; exact deterministic full replay",
            "Internal temporal reconstruction consistency",
        ],
        [
            "Q8",
            (
                "Does retained evolved state/history improve withheld "
                "next-slice reconstruction?"
            ),
            "Yes in this synthetic benchmark",
            "History-dependent synthetic reconstruction",
        ],
        [
            "Q9",
            (
                "Does A -> B versus B -> A change the response to the same "
                "subsequent event C?"
            ),
            str(bool(require(
                temporal,
                "operator_evolution",
                "history_changes_response_to_c",
            ))),
            "History-conditioned response to the same subsequent event",
        ],
    ]
    headers = ["Test", "Question", "Result", "Supported claim"]
    csv_path = TABLES / "table_benchmark_summary.csv"
    tex_path = TABLES / "table_benchmark_summary.tex"
    write_csv(csv_path, headers, rows)
    write_latex_table(
        tex_path, headers, rows,
        caption="Summary of computational benchmark questions and supported claims.",
        label="tab:benchmark-summary",
        column_spec=r"p{0.11\textwidth}p{0.35\textwidth}p{0.18\textwidth}p{0.23\textwidth}",
    )
    return {
        "csv": str(csv_path.relative_to(ROOT)),
        "tex": str(tex_path.relative_to(ROOT)),
    }


def run() -> dict[str, Any]:
    physical = load_json("physical_history_integration_v1.json")
    separability = load_json("separability_benchmark_v1.json")
    q6 = load_json("q6_lyapunov_like_v1.json")
    temporal = load_json("temporal_image_reconstruction_v1.json")

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    assets = {
        "generator_version": GENERATOR_VERSION,
        "claim_scope": "computational_model_only",
        "source_files": [
            "benchmark_results/physical_history_integration_v1.json",
            "benchmark_results/separability_benchmark_v1.json",
            "benchmark_results/q6_lyapunov_like_v1.json",
            "benchmark_results/temporal_image_reconstruction_v1.json",
        ],
        "figures": {
            "global_redistribution": figure_global_redistribution(physical),
            "path_dependence": figure_path_dependence(physical),
            "repeated_event": figure_repeated_event(physical),
            "separability": figure_separability(separability),
            "q6": figure_q6(q6),
            "temporal_reconstruction": figure_temporal_reconstruction(
                temporal
            ),
            "operator_evolution": figure_operator_evolution(
                temporal
            ),
        },
        "tables": {
            "physical_history": table_physical_history(physical),
            "temporal_reconstruction": table_temporal_reconstruction(
                temporal
            ),
            "operator_evolution": table_operator_evolution(
                temporal
            ),
            "claim_boundaries": table_claim_boundaries(),
            "benchmark_summary": table_benchmark_summary(
                physical,
                separability,
                q6,
                temporal,
            ),
        },
    }

    manifest = PAPER / "publication_asset_manifest.json"
    manifest.write_text(
        json.dumps(assets, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    assets["manifest"] = str(manifest.relative_to(ROOT))
    return assets


def main() -> None:
    assets = run()
    print(json.dumps(assets, indent=2, sort_keys=True))
    print("\nPublication assets generated in:")
    print(PAPER.resolve())


if __name__ == "__main__":
    main()
