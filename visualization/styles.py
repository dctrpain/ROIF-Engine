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
