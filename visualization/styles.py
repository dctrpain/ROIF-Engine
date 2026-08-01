from __future__ import annotations

"""
Central visual style definitions for NetworkPlotter.

Geometry mode:
    neutral reference and current geometry styles.

Force mode:
    line width = absolute transmitted force;
    line color = normalized force stimulus.

Material modes:
    line color = selected normalized material-state variable;
    line width = constant unless the element has failed.
"""

REFERENCE_ELEMENT_STYLE = {
    "color": "#8A8A8A",
    "linestyle": "--",
    "linewidth": 1.2,
    "alpha": 0.65,
    "zorder": 1,
}

CURRENT_ELEMENT_STYLE = {
    "color": "#1F4E79",
    "linestyle": "-",
    "linewidth": 2.0,
    "alpha": 1.0,
    "zorder": 2,
}

FAILED_ELEMENT_STYLE = {
    "color": "#B22222",
    "linestyle": ":",
    "linewidth": 2.4,
    "alpha": 1.0,
    "zorder": 3,
}

FIXED_NODE_STYLE = {
    "color": "#202020",
    "marker": "s",
    "s": 75,
    "zorder": 5,
}

FREE_NODE_STYLE = {
    "color": "#F28E2B",
    "marker": "o",
    "s": 75,
    "zorder": 5,
}

NODE_LABEL_STYLE = {
    "fontsize": 10,
    "color": "#202020",
}

GRID_STYLE = {
    "alpha": 0.35,
    "linewidth": 0.8,
}

FORCE_COLORMAP = "viridis"
FORCE_MIN_LINEWIDTH = 1.2
FORCE_MAX_LINEWIDTH = 5.0
FORCE_COLORBAR_LABEL = "Normalized force stimulus"

MATERIAL_MODE_STYLES = {
    "damage": {
        "cmap": "inferno",
        "label": "Damage",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    "fatigue": {
        "cmap": "plasma",
        "label": "Fatigue",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    "integrity": {
        "cmap": "viridis",
        "label": "Integrity",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    "remodeling": {
        "cmap": "cividis",
        "label": "Remodeling",
        "vmin": 0.0,
        "vmax": 1.0,
    },
}

# ============================================================
# ROIF Viewer v3.0 — mechanical-role styles
# ============================================================

ELEMENT_ROLE_STYLES = {
    "rigid": {
        "color": "#303030",
        "linestyle": "-",
        "linewidth": 5.0,
        "alpha": 1.0,
        "zorder": 4,
    },
    "tension": {
        "color": "#1F77B4",
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 1.0,
        "zorder": 3,
    },
    "active_tension": {
        "color": "#D62728",
        "linestyle": "-",
        "linewidth": 2.6,
        "alpha": 1.0,
        "zorder": 4,
    },
    "compression": {
        "color": "#9467BD",
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 3,
    },
    "generic": {
        **CURRENT_ELEMENT_STYLE,
    },
    "disabled": {
        "color": "#9A9A9A",
        "linestyle": "--",
        "linewidth": 1.4,
        "alpha": 0.35,
        "zorder": 1,
    },
    "failed": {
        **FAILED_ELEMENT_STYLE,
    },
}


ELEMENT_ROLE_LABELS = {
    "rigid": "Rigid / strut",
    "tension": "Tension-only",
    "active_tension": "Active tension",
    "compression": "Compression-only",
    "generic": "Generic axial",
    "disabled": "Disabled",
    "failed": "Failed",
}


def element_role_style(
    role: object,
) -> dict[str, object]:
    """
    Return an independent Matplotlib style for a viewer element role.

    role may be either:

    - an ElementVisualRole enum value;
    - a string such as ``"rigid"`` or ``"tension"``.

    Unknown values safely fall back to the generic axial style.
    """

    value = getattr(
        role,
        "value",
        role,
    )
    key = str(value).lower()

    style = ELEMENT_ROLE_STYLES.get(
        key,
        ELEMENT_ROLE_STYLES["generic"],
    )

    return dict(style)


def element_role_label(
    role: object,
) -> str:
    """Return a human-readable label for a viewer element role."""

    value = getattr(
        role,
        "value",
        role,
    )
    key = str(value).lower()

    return ELEMENT_ROLE_LABELS.get(
        key,
        ELEMENT_ROLE_LABELS["generic"],
    )