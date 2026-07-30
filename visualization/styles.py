from __future__ import annotations

"""
Central visual style definitions for NetworkPlotter.

These values have no mechanical meaning. They only make geometry
categories visually consistent and prevent Matplotlib from assigning
a different automatic color to every element.
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
