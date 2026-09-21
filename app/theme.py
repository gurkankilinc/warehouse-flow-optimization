"""Industrial visual language, and the one place the building is drawn.

The palette is the one operations screens use -- near-black ground, steel grey
structure, amber for attention, red for a miss and green for a save. Numbers are
monospaced so columns of figures line up.

``draw_floor_plan`` is deliberately the *only* function that renders the
warehouse. The heat map, the putaway recommendation and the order journey all
call it, so the building looks identical wherever it appears and a change to the
layout shows up everywhere at once.

One layout rule matters more than any colour: **nothing here may be wider than
its container.** Figures are created at modest widths and always rendered with
``use_container_width=True``. A figure wider than the page gives Streamlit a
horizontal scrollbar, and because its sidebar is ``position: fixed`` the sidebar
then paints at both scroll extremes and appears duplicated -- which is exactly
how this dashboard became unusable the first time.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from config import CONFIG, WarehouseConfig

__all__ = [
    "PALETTE",
    "CSS",
    "apply_matplotlib_style",
    "new_figure",
    "draw_floor_plan",
    "rack_cell_demand",
    "distance_bands",
]

PALETTE = {
    "ground": "#0d1117",
    "panel": "#161b22",
    "grid": "#21262d",
    "steel": "#8b98a5",
    "steel_dim": "#4a5560",
    "text": "#e6edf3",
    "amber": "#f0a12e",
    "amber_dim": "#7a5314",
    "red": "#e5484d",
    "green": "#3fb950",
    "blue": "#4493f8",
    "cold": "#3aa8c1",
}

#: Heat scales. Both run dark -> amber so they read on the dark ground.
HEAT_CMAP = "inferno"
DIVERGING_CMAP = "RdBu_r"


CSS = f"""
<style>
  /* ---- the overflow guard -------------------------------------------
     A single element wider than the viewport gives the page a horizontal
     scrollbar, and Streamlit's fixed-position sidebar then renders at both
     scroll extremes -- the "left side drawn twice" bug. These rules make
     that impossible rather than merely unlikely.                        */
  [data-testid="stAppViewContainer"] {{ overflow-x: hidden; }}
  [data-testid="stMain"] {{ overflow-x: hidden; }}
  .block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
  img, svg, canvas {{ max-width: 100%; height: auto; }}

  /* ---- industrial surface ---- */
  [data-testid="stAppViewContainer"] {{ background: {PALETTE['ground']}; }}
  [data-testid="stSidebar"] {{
      background: {PALETTE['panel']};
      border-right: 1px solid {PALETTE['grid']};
  }}
  h1, h2, h3, h4 {{ color: {PALETTE['text']}; letter-spacing: -0.01em; }}
  h1 {{ font-weight: 700; }}

  /* Section rule that reads like a drawing border rather than a web divider */
  hr {{ border-color: {PALETTE['grid']}; }}

  /* ---- metrics: monospaced numerals, amber accent ---- */
  [data-testid="stMetricValue"] {{
      font-family: "SFMono-Regular", "JetBrains Mono", Consolas, monospace;
      color: {PALETTE['amber']};
      font-size: 1.6rem;
  }}
  [data-testid="stMetricLabel"] {{
      color: {PALETTE['steel']};
      text-transform: uppercase;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
  }}

  /* ---- tabs as control-panel buttons ----
     Selector covers both markups Streamlit has shipped: older builds tag tabs
     with data-baseweb, current ones only with role="tab". Matching one alone
     silently leaves the tabs unstyled. `flex-wrap` is what stops six tab
     labels from forcing a horizontal scrollbar on a narrow screen.        */
  .stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {{
      gap: 2px;
      border-bottom: 1px solid {PALETTE['grid']};
      flex-wrap: wrap;
  }}
  .stTabs [role="tab"], .stTabs [data-baseweb="tab"] {{
      background: transparent;
      color: {PALETTE['steel']};
      border-radius: 3px 3px 0 0;
      padding: 0.5rem 0.9rem;
      font-size: 0.88rem;
  }}
  .stTabs [role="tab"][aria-selected="true"],
  .stTabs [data-baseweb="tab"][aria-selected="true"] {{
      color: {PALETTE['amber']};
      border-bottom: 2px solid {PALETTE['amber']};
  }}

  /* ---- callout blocks used on the overview tab ---- */
  .wf-card {{
      background: {PALETTE['panel']};
      border: 1px solid {PALETTE['grid']};
      border-left: 3px solid {PALETTE['amber']};
      border-radius: 4px;
      padding: 0.9rem 1.1rem;
      margin-bottom: 0.8rem;
  }}
  .wf-card h4 {{ margin: 0 0 0.4rem 0; font-size: 0.95rem; color: {PALETTE['amber']}; }}
  .wf-card p, .wf-card li {{ color: {PALETTE['text']}; font-size: 0.9rem; margin: 0.25rem 0; }}
  .wf-card.negative {{ border-left-color: {PALETTE['red']}; }}
  .wf-card.negative h4 {{ color: {PALETTE['red']}; }}
  .wf-card.positive {{ border-left-color: {PALETTE['green']}; }}
  .wf-card.positive h4 {{ color: {PALETTE['green']}; }}

  .wf-tag {{
      display: inline-block;
      font-family: monospace;
      font-size: 0.72rem;
      padding: 0.12rem 0.45rem;
      border-radius: 3px;
      border: 1px solid {PALETTE['steel_dim']};
      color: {PALETTE['steel']};
      margin-right: 0.35rem;
  }}
</style>
"""


def apply_matplotlib_style() -> None:
    """Make every figure match the page instead of fighting it."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["panel"],
            "axes.facecolor": PALETTE["panel"],
            "savefig.facecolor": PALETTE["panel"],
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["steel"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.titlecolor": PALETTE["text"],
            "xtick.color": PALETTE["steel"],
            "ytick.color": PALETTE["steel"],
            "grid.color": PALETTE["grid"],
            "axes.grid": True,
            "grid.alpha": 0.4,
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.autolayout": True,
        }
    )


def new_figure(width: float = 7.0, height: float = 4.0, **kwargs):
    """Create a figure that cannot overflow the page.

    Width is capped deliberately. Anything wider than roughly the content
    column reintroduces the horizontal scrollbar that duplicates the sidebar.
    """
    width = min(width, 9.0)
    return plt.subplots(figsize=(width, height), **kwargs)


# ----------------------------------------------------------------------
# The building
# ----------------------------------------------------------------------


def draw_floor_plan(
    ax,
    cfg: WarehouseConfig | None = None,
    *,
    cell_values: pd.DataFrame | None = None,
    cmap: str = HEAT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    show_cold_zone: bool = True,
    show_labels: bool = True,
):
    """Draw the warehouse the way a facility drawing would.

    Racks are blocks, aisles are the gaps between them, cross aisles are
    labelled corridors and the dock is a marked opening on the front wall --
    rather than the cloud of anonymous dots this replaced.

    ``cell_values`` optionally colours each rack cell; it must carry columns
    ``aisle``, ``side``, ``bay`` and ``value`` (see :func:`rack_cell_demand`).
    """
    cfg = cfg or CONFIG.warehouse
    rack_w = cfg.aisle_width * 0.36
    offset = cfg.aisle_width * 0.30
    bay_h = cfg.bay_depth * 0.92

    mappable = None
    if cell_values is not None and not cell_values.empty:
        values = cell_values["value"].to_numpy(dtype=float)
        vmin = float(values.min()) if vmin is None else vmin
        vmax = float(values.max()) if vmax is None else vmax
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colormap = mpl.colormaps[cmap]
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        lookup = {
            (int(r.aisle), str(r.side), int(r.bay)): float(r.value)
            for r in cell_values.itertuples(index=False)
        }
    else:
        lookup = {}
        colormap = None
        norm = None

    # --- cold zone backdrop -------------------------------------------
    if show_cold_zone and cfg.cold_zone_aisles:
        cold = sorted(cfg.cold_zone_aisles)
        x0 = cold[0] * cfg.aisle_width - cfg.aisle_width * 0.55
        x1 = cold[-1] * cfg.aisle_width + cfg.aisle_width * 0.55
        ax.add_patch(
            Rectangle(
                (x0, -1.2),
                x1 - x0,
                cfg.aisle_length + 2.4,
                facecolor=PALETTE["cold"],
                alpha=0.10,
                edgecolor=PALETTE["cold"],
                linestyle="--",
                linewidth=0.8,
                zorder=0,
            )
        )
        if show_labels:
            ax.text(
                (x0 + x1) / 2,
                cfg.aisle_length + 1.6,
                "COLD ZONE",
                ha="center",
                va="bottom",
                fontsize=7,
                color=PALETTE["cold"],
                weight="bold",
            )

    # --- rack blocks ---------------------------------------------------
    for aisle in range(cfg.n_aisles):
        x_centre = aisle * cfg.aisle_width
        for side, sign in (("L", -1), ("R", 1)):
            x = x_centre + sign * offset - rack_w / 2
            for bay in range(cfg.bays_per_aisle):
                y = cfg.cross_aisle_offset + bay * cfg.bay_depth
                value = lookup.get((aisle, side, bay))
                if value is None:
                    face, alpha = PALETTE["steel_dim"], 0.35
                else:
                    face, alpha = colormap(norm(value)), 1.0
                ax.add_patch(
                    Rectangle(
                        (x, y - bay_h / 2),
                        rack_w,
                        bay_h,
                        facecolor=face,
                        edgecolor="none",
                        alpha=alpha,
                        zorder=2,
                    )
                )

    # --- cross aisles ---------------------------------------------------
    span_x = (-cfg.aisle_width, (cfg.n_aisles - 1) * cfg.aisle_width + cfg.aisle_width)
    for i, y in enumerate(cfg.cross_aisle_ys):
        ax.plot(span_x, [y, y], color=PALETTE["amber"], lw=1.0, ls="--", alpha=0.55, zorder=3)
        if show_labels:
            label = "FRONT" if i == 0 else ("BACK" if i == len(cfg.cross_aisle_ys) - 1 else "MID")
            ax.text(
                span_x[1] + 0.6,
                y,
                label,
                fontsize=6.5,
                color=PALETTE["amber"],
                va="center",
                ha="left",
                alpha=0.85,
            )

    # --- dock and goods-in ----------------------------------------------
    ax.add_patch(
        Rectangle(
            (cfg.dock_x - 1.5, -2.6),
            3.0,
            1.6,
            facecolor=PALETTE["blue"],
            edgecolor="none",
            zorder=4,
        )
    )
    if show_labels:
        ax.text(
            cfg.dock_x,
            -3.1,
            "DOCK",
            ha="center",
            va="top",
            fontsize=7,
            color=PALETTE["blue"],
            weight="bold",
        )

    ax.set_xlim(span_x[0] - 1, span_x[1] + 5)
    ax.set_ylim(-4.5, cfg.aisle_length + 4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return mappable


# ----------------------------------------------------------------------
# Aggregations the views share
# ----------------------------------------------------------------------


def rack_cell_demand(
    slots: pd.DataFrame,
    assignment: pd.DataFrame,
    visits: pd.Series,
    *,
    as_rank: bool = True,
) -> pd.DataFrame:
    """Pick frequency per rack cell (a bay's three levels combined).

    ``as_rank`` converts to percentile rank. Pick frequency follows a Zipf law,
    so on a raw scale a handful of SKUs take the whole colour range and every
    other cell collapses into the first swatch -- which is why the original
    heat map showed nothing. Ranking makes the scale uniform by construction.
    """
    merged = slots.merge(assignment[["sku_id", "slot_id"]], on="slot_id", how="left")
    merged["picks"] = merged["sku_id"].map(visits).fillna(0.0)

    cells = (
        merged.groupby(["aisle", "side", "bay"], as_index=False)["picks"]
        .sum()
        .rename(columns={"picks": "value"})
    )
    if as_rank:
        cells["value"] = cells["value"].rank(pct=True) * 100.0
    return cells


def distance_bands(
    slots: pd.DataFrame,
    assignment: pd.DataFrame,
    visits: pd.Series,
    *,
    band_m: float = 10.0,
) -> pd.DataFrame:
    """Share of total picking work falling in each distance band from the dock.

    The heat map shows *where* the demand sits; this shows *how much*. It is the
    one-line summary of whether a layout is any good, and it stays legible where
    a heat map does not.
    """
    merged = slots.merge(assignment[["sku_id", "slot_id"]], on="slot_id", how="left")
    merged["picks"] = merged["sku_id"].map(visits).fillna(0.0)

    edges = np.arange(0, merged["dock_distance"].max() + band_m, band_m)
    merged["band"] = pd.cut(merged["dock_distance"], bins=edges, right=False)

    grouped = merged.groupby("band", observed=True)["picks"].sum()
    total = max(grouped.sum(), 1.0)
    return pd.DataFrame(
        {
            "band": [f"{int(i.left)}-{int(i.right)}" for i in grouped.index],
            "picks": grouped.to_numpy(),
            "share_pct": 100.0 * grouped.to_numpy() / total,
        }
    )
