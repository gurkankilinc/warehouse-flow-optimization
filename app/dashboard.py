"""Warehouse Flow Optimization — interactive dashboard.

Run with::

    streamlit run app/dashboard.py

Six views, one per question the project answers:

  Overview       what problem, what goal, what solution, what to do about it
  Warehouse      the facility in detail, and where the demand actually sits
  Optimization   what is optimized, on what horizon, by what method, for what gain
  Journey        one order's physical trip through the building, animated
  Putaway        a pallet arrives — which slot, and why
  Results        the measured evidence, including what did not work

Everything is read from artefacts the pipeline produced. The dashboard never
trains a model: if the cache is missing it says which command to run rather than
silently spending two minutes fitting three models on page load.

**Layout rule:** no element may exceed its container width. Figures are built
through ``theme.new_figure`` (which caps width) and rendered with
``use_container_width=True``; side-by-side comparisons use ``st.columns`` rather
than one double-width figure. A single over-wide element gives the page a
horizontal scrollbar, and Streamlit's fixed-position sidebar then paints at both
scroll extremes — the "left side drawn twice" bug this dashboard started with.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG, REPORTS_DIR  # noqa: E402
from src.experiments.pipeline import CONTEXT_CACHE  # noqa: E402
from src.policies.routing import build_distance_matrix, route_nn_2opt, route_s_shape  # noqa: E402
from src.policies.slotting import abc_xyz_classes  # noqa: E402
from src.simulation.generate import load_dataset  # noqa: E402

from app.i18n import LANGUAGES, t  # noqa: E402
from app.svg_map import render_journey_svg  # noqa: E402
from app.theme import (  # noqa: E402
    CSS,
    HEAT_CMAP,
    PALETTE,
    apply_matplotlib_style,
    distance_bands,
    draw_floor_plan,
    new_figure,
    rack_cell_demand,
)

st.set_page_config(
    page_title="Warehouse Flow Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_tables():
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_context():
    import joblib

    if not CONTEXT_CACHE.exists():
        return None
    try:
        context = joblib.load(CONTEXT_CACHE)
        if not context.tables:
            context.tables = get_tables()
        return context
    except Exception:  # noqa: BLE001 - a stale cache must not break the page
        return None


@st.cache_data(show_spinner=False)
def get_report(name: str):
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def get_visits() -> pd.Series:
    """Pick count per SKU over the simulated window."""
    return get_tables()["order_lines"].groupby("sku_id").size()


@st.cache_data(show_spinner=False)
def get_abc_xyz() -> pd.DataFrame:
    """ABC/XYZ classes, cached.

    This groups 1.3 million daily-demand rows. Recomputing it on every rerun
    would make the urgency slider on the placement tab unusable -- the whole
    point of that control is that the layout moves as you drag it.
    """
    tables = get_tables()
    return abc_xyz_classes(tables["daily_demand"], tables["sku_catalog"])


def card(title: str, body_md: str, kind: str = "accent") -> None:
    """A bordered callout whose body is real Markdown.

    An earlier version wrapped the body in a raw ``<div>``, which meant Streamlit
    never ran Markdown over it and ``**bold**`` reached the page as literal
    asterisks. Using a native bordered container keeps the styling and lets the
    body render properly.
    """
    colour = {
        "accent": PALETTE["amber"],
        "negative": PALETTE["red"],
        "positive": PALETTE["green"],
    }.get(kind, PALETTE["amber"])

    with st.container(border=True):
        st.markdown(
            f"<div style='color:{colour};font-weight:700;font-size:0.95rem;"
            f"margin-bottom:0.35rem'>{title}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(body_md)


# ----------------------------------------------------------------------
# Diagrams
# ----------------------------------------------------------------------


def draw_layer_diagram(lang: str, gains: dict[str, float]):
    """The three decision layers, their inputs, methods and measured gain."""
    fig, ax = new_figure(8.4, 4.6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    tr = lang == "tr"
    layers = [
        {
            "y": 5.2,
            "title": "TAKTİK  ·  haftalık" if tr else "TACTICAL  ·  weekly",
            "decision": "Hangi ürün hangi rafta?" if tr else "Which SKU in which slot?",
            "input": "Talep tahmini × aciliyet" if tr else "Forecast demand x urgency",
            "method": "Kısıtlı atama" if tr else "Constrained assignment",
            "gain": gains.get("slotting_only"),
            "colour": PALETTE["blue"],
        },
        {
            "y": 3.2,
            "title": "OPERASYONEL  ·  dakikalar" if tr else "OPERATIONAL  ·  minutes",
            "decision": "Sıradaki sipariş hangisi?" if tr else "Which order is picked next?",
            "input": "Teslim süresi, tahmini süre" if tr else "Deadline, predicted pick time",
            "method": "Triage (Moore-Hodgson)",
            "gain": gains.get("dispatch_only"),
            "colour": PALETTE["amber"],
        },
        {
            "y": 1.2,
            "title": "ANLIK  ·  saniyeler" if tr else "IMMEDIATE  ·  seconds",
            "decision": "Duraklar hangi sırayla?" if tr else "In what sequence are stops visited?",
            "input": "Raf koordinatları" if tr else "Slot coordinates",
            "method": "En yakın komşu + 2-opt" if tr else "Nearest neighbour + 2-opt",
            "gain": gains.get("routing_only"),
            "colour": PALETTE["green"],
        },
    ]

    for layer in layers:
        y = layer["y"]
        ax.add_patch(
            FancyBboxPatch(
                (0.3, y - 0.72),
                9.4,
                1.5,
                boxstyle="round,pad=0.04",
                facecolor=PALETTE["ground"],
                edgecolor=layer["colour"],
                linewidth=1.4,
            )
        )
        ax.text(0.6, y + 0.52, layer["title"], color=layer["colour"], fontsize=9, weight="bold")
        ax.text(0.6, y + 0.06, layer["decision"], color=PALETTE["text"], fontsize=9)
        ax.text(
            0.6, y - 0.42,
            f"← {layer['input']}      ⚙ {layer['method']}",
            color=PALETTE["steel"], fontsize=7.5,
        )
        if layer["gain"] is not None:
            ax.text(
                9.4, y + 0.05,
                f"{layer['gain']:+.1f} pp",
                color=layer["colour"], fontsize=13, weight="bold", ha="right",
                family="monospace",
            )
            ax.text(
                9.4, y - 0.42,
                "zamanında sevk" if tr else "on-time rate",
                color=PALETTE["steel"], fontsize=7, ha="right",
            )

    for y0, y1 in ((5.2 - 0.75, 3.2 + 0.8), (3.2 - 0.75, 1.2 + 0.8)):
        ax.annotate(
            "", xy=(5.0, y1), xytext=(5.0, y0),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["steel_dim"], lw=1.2),
        )

    ax.text(
        5.0, 6.85,
        "Talep tahmini (ML-1) taktik katmanı besler" if tr
        else "The demand forecast (ML-1) feeds the tactical layer",
        color=PALETTE["steel"], fontsize=8, ha="center", style="italic",
    )
    return fig


def draw_order_stages(lang: str):
    """The physical stages an order passes through, and where the deadline comes from."""
    fig, ax = new_figure(8.4, 2.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    tr = lang == "tr"
    stages = (
        ["Sipariş\ngelir", "Kuyrukta\nbekler", "Tura\natanır", "Toplanır", "Hazırlanır", "Kamyona\nyüklenir"]
        if tr
        else ["Order\narrives", "Waits in\nqueue", "Assigned\nto a tour", "Picked", "Staged", "Loaded on\ntruck"]
    )
    colours = [PALETTE["steel"]] * 2 + [PALETTE["amber"]] * 3 + [PALETTE["blue"]]

    width = 1.42
    for i, (label, colour) in enumerate(zip(stages, colours)):
        x = 0.25 + i * 1.62
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.15), width, 0.95,
                boxstyle="round,pad=0.03",
                facecolor=PALETTE["ground"], edgecolor=colour, linewidth=1.2,
            )
        )
        ax.text(x + width / 2, 1.62, label, color=PALETTE["text"], fontsize=7.5,
                ha="center", va="center")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + width + 0.18, 1.62), xytext=(x + width + 0.02, 1.62),
                        arrowprops=dict(arrowstyle="-|>", color=PALETTE["steel_dim"], lw=1.1))

    ax.annotate(
        "", xy=(9.05, 1.1), xytext=(9.05, 0.55),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["red"], lw=1.2),
    )
    ax.text(
        9.0, 0.3,
        "son hazırlık anı = kamyon kalkışı − hazırlık tamponu" if tr
        else "staging deadline = truck departure − staging buffer",
        color=PALETTE["red"], fontsize=7.5, ha="right",
    )
    return fig


def draw_timeline(order_row, lang: str):
    """Where this order's deadline sits between its arrival and its truck."""
    fig, ax = new_figure(8.4, 1.9)
    tr = lang == "tr"

    created = float(order_row["created_h"])
    deadline = float(order_row["deadline_h"])
    departure = float(order_row["truck_departure_h"])
    span = max(departure - created, 1e-6)

    ax.set_xlim(-0.06 * span, span * 1.12)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([0, span], [0.55, 0.55], color=PALETTE["steel_dim"], lw=3, solid_capstyle="round")
    ax.plot([0, deadline - created], [0.55, 0.55], color=PALETTE["amber"], lw=3,
            solid_capstyle="round")

    marks = [
        (0.0, "Sipariş gelir" if tr else "Order arrives", PALETTE["green"]),
        (deadline - created, "Son hazırlık anı" if tr else "Staging deadline", PALETTE["red"]),
        (span, "Kamyon kalkar" if tr else "Truck departs", PALETTE["blue"]),
    ]
    for x, label, colour in marks:
        ax.plot([x], [0.55], marker="o", ms=9, color=colour, zorder=3)
        ax.text(x, 0.80, label, color=colour, fontsize=8, ha="center")
        ax.text(x, 0.30, f"+{x:.1f} h", color=PALETTE["steel"], fontsize=7.5,
                ha="center", family="monospace")

    ax.text(
        span * 0.5, 0.05,
        f"{'Toplam süre' if tr else 'Total window'}: {span:.1f} h",
        color=PALETTE["steel"], fontsize=7.5, ha="center",
    )
    return fig


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------


def view_overview(lang: str, tables, context):
    raw = get_report("scenario_raw.csv")

    st.subheader(t("ov_problem_h", lang))
    st.markdown(t("ov_problem", lang))

    left, right = st.columns(2)
    with left:
        card(t("ov_goal_h", lang), t("ov_goal", lang))
    with right:
        card(t("ov_solution_h", lang), t("ov_solution", lang))

    if raw is not None and "baseline" in set(raw["scenario"]):
        means = raw.groupby("scenario")["on_time_rate"].mean()
        dist = raw.groupby("scenario")["distance_per_line_m"].mean()
        base_miss = 100 * (1 - means["baseline"])
        best = (1 - means).idxmin()
        best_miss = 100 * (1 - means[best])

        st.subheader(t("ov_headline_h", lang))
        cols = st.columns(3)
        cols[0].metric(f"{t('rs_missed', lang)} — {t('rs_baseline', lang)}", f"{base_miss:.2f}%")
        cols[1].metric(
            f"{t('rs_missed', lang)} — {t('rs_best', lang)}",
            f"{best_miss:.2f}%",
            f"{100 * (best_miss / base_miss - 1):.0f}%",
            delta_color="inverse",
        )
        cols[2].metric(
            t("rs_distance", lang),
            f"{dist[best]:.2f} m",
            f"{100 * (dist[best] / dist['baseline'] - 1):.1f}%",
            delta_color="inverse",
        )

    st.divider()
    st.subheader(t("ov_reco_h", lang))
    st.markdown(t("ov_reco", lang))

    st.divider()
    card(t("ov_honest_h", lang), t("ov_honest", lang), kind="negative")


def view_warehouse(lang: str, tables, context):
    cfg = CONFIG.warehouse
    slots = tables["warehouse_slots"]
    catalog = tables["sku_catalog"]
    tr = lang == "tr"

    # ---------------- specification ----------------
    st.subheader(t("wh_specs_h", lang))
    spec_rows = [
        ("Koridor sayısı" if tr else "Picking aisles", f"{cfg.n_aisles}"),
        ("Koridor başına göz" if tr else "Bays per aisle", f"{cfg.bays_per_aisle}"),
        ("Raf yüzü (koridor başına)" if tr else "Rack faces per aisle", f"{cfg.sides}"),
        ("Kat sayısı" if tr else "Levels", f"{cfg.levels}"),
        ("Toplam raf gözü" if tr else "Total storage slots", f"{cfg.n_slots:,}"),
        ("Ürün çeşidi (SKU)" if tr else "Distinct SKUs", f"{len(catalog):,}"),
        ("Doluluk" if tr else "Slot occupancy", f"{100 * len(catalog) / cfg.n_slots:.0f}%"),
        ("Koridor genişliği" if tr else "Aisle width", f"{cfg.aisle_width:.1f} m"),
        ("Göz derinliği" if tr else "Bay depth", f"{cfg.bay_depth:.1f} m"),
        ("Koridor uzunluğu" if tr else "Aisle length", f"{cfg.aisle_length:.1f} m"),
        ("Bina genişliği" if tr else "Building width",
         f"{(cfg.n_aisles - 1) * cfg.aisle_width:.1f} m"),
        ("Çapraz koridorlar" if tr else "Cross aisles",
         ", ".join(f"y={v:.1f} m" for v in cfg.cross_aisle_ys)),
        ("Dock'a en kısa / en uzun" if tr else "Nearest / farthest slot",
         f"{slots.dock_distance.min():.1f} / {slots.dock_distance.max():.1f} m"),
    ]
    ops_rows = [
        ("Toplayıcı" if tr else "Pickers", f"{CONFIG.picking.n_pickers}"),
        ("Mal kabul personeli" if tr else "Putaway crew", f"{CONFIG.picking.n_putaway_workers}"),
        ("Vardiya" if tr else "Shift",
         f"{CONFIG.simulation.shift_start_hour:.0f}:00 – {CONFIG.simulation.shift_end_hour:.0f}:00"),
        ("Yürüme hızı" if tr else "Walking speed", f"{CONFIG.picking.walking_speed_mps:.2f} m/s"),
        ("Araba kapasitesi" if tr else "Cart capacity",
         f"{CONFIG.picking.max_lines_per_tour} {'kalem' if tr else 'lines'} / "
         f"{CONFIG.picking.max_volume_per_tour_l:.0f} L"),
        ("Günlük kamyon kalkışı" if tr else "Truck departures per day",
         ", ".join(f"{h:.0f}:00" for h in CONFIG.truck.departure_hours)),
        ("Dock kapısı" if tr else "Dock doors", f"{CONFIG.truck.n_docks}"),
        ("Hazırlık tamponu" if tr else "Staging buffer", f"{CONFIG.truck.staging_buffer_h:.1f} h"),
        ("Kamyon kapasitesi" if tr else "Truck capacity",
         f"{CONFIG.truck.truck_capacity_orders} {'sipariş' if tr else 'orders'}"),
        ("Servis seviyeleri" if tr else "Service levels",
         ", ".join(CONFIG.demand.service_levels)),
    ]
    left, right = st.columns(2)
    left.dataframe(
        pd.DataFrame(spec_rows, columns=["", "Değer" if tr else "Value"]),
        use_container_width=True, hide_index=True,
    )
    right.dataframe(
        pd.DataFrame(ops_rows, columns=["", "Değer" if tr else "Value"]),
        use_container_width=True, hide_index=True,
    )

    # ---------------- zones ----------------
    st.subheader(t("wh_zones_h", lang))
    n_cold_slots = int(slots.is_cold.sum())
    n_ground = int(slots.is_ground.sum())
    n_cold_sku = int(catalog.is_cold.sum())
    n_heavy_sku = int(catalog.is_heavy.sum())
    zone_rows = [
        (
            "Soğuk zincir" if tr else "Cold chain",
            f"{'Koridor' if tr else 'Aisles'} {min(cfg.cold_zone_aisles)}–{max(cfg.cold_zone_aisles)}",
            f"{n_cold_slots:,} {'göz' if tr else 'slots'}",
            f"{n_cold_sku:,} SKU",
            ("Soğuk ürünler yalnızca burada; ortam ürünleri buraya giremez"
             if tr else "Cold SKUs must be here; ambient SKUs may not"),
        ),
        (
            "Ağır ürün" if tr else "Heavy goods",
            "Zemin kat" if tr else "Ground level only",
            f"{n_ground:,} {'göz' if tr else 'slots'}",
            f"{n_heavy_sku:,} SKU",
            ("Ergonomi: ağır ürün üst katlara çıkmaz"
             if tr else "Ergonomics: heavy SKUs never go above floor level"),
        ),
    ]
    st.dataframe(
        pd.DataFrame(
            zone_rows,
            columns=(
                ["Bölge", "Konum", "Kapasite", "Bağlayan SKU", "Kısıt"] if tr
                else ["Zone", "Location", "Capacity", "SKUs bound", "Constraint"]
            ),
        ),
        use_container_width=True, hide_index=True,
    )

    if context is None:
        st.warning(t("artifacts_missing", lang))
        return

    # ---------------- heat map ----------------
    st.divider()
    st.subheader(t("wh_heat_h", lang))
    st.caption(t("wh_heat_help", lang))

    visits = get_visits()
    base_cells = rack_cell_demand(slots, context.abc_assignment, visits)
    opt_cells = rack_cell_demand(slots, context.urgency_assignment, visits)

    left, right = st.columns(2)
    for column, (cells, title) in zip(
        (left, right),
        [
            (base_cells, "Taban: geçmişe dayalı ABC" if tr else "Baseline: trailing ABC"),
            (opt_cells, "Optimize: tahmin × aciliyet" if tr else "Optimised: forecast x urgency"),
        ],
    ):
        with column:
            fig, ax = new_figure(5.2, 4.4)
            mappable = draw_floor_plan(ax, cfg, cell_values=cells, cmap=HEAT_CMAP, vmin=0, vmax=100)
            ax.set_title(title, fontsize=9)
            if mappable is not None:
                bar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.02)
                bar.set_label("yüzdelik dilim" if tr else "percentile", fontsize=7.5)
                bar.ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=True)

    # ---------------- distance bands ----------------
    st.subheader(t("wh_bands_h", lang))
    st.caption(t("wh_bands_help", lang))

    base_bands = distance_bands(slots, context.abc_assignment, visits)
    opt_bands = distance_bands(slots, context.urgency_assignment, visits)

    fig, ax = new_figure(8.4, 3.4)
    idx = np.arange(len(base_bands))
    ax.bar(idx - 0.2, base_bands["share_pct"], 0.4,
           label="Taban" if tr else "Baseline", color=PALETTE["steel"])
    ax.bar(idx + 0.2, opt_bands["share_pct"], 0.4,
           label="Optimize" if tr else "Optimised", color=PALETTE["amber"])
    ax.set_xticks(idx)
    ax.set_xticklabels(base_bands["band"], fontsize=7.5)
    ax.set_xlabel("dock'tan mesafe (m)" if tr else "distance from dock (m)")
    ax.set_ylabel("toplama işi payı (%)" if tr else "share of picking work (%)")
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

    near = base_bands["share_pct"].iloc[:2].sum(), opt_bands["share_pct"].iloc[:2].sum()
    st.caption(
        (f"İlk 20 m içindeki toplama işi payı — taban %{near[0]:.1f}, optimize %{near[1]:.1f}."
         if tr else
         f"Share of picking work within the first 20 m — baseline {near[0]:.1f}%, "
         f"optimised {near[1]:.1f}%.")
    )

    # ---------------- ABC x XYZ ----------------
    st.divider()
    st.subheader(t("wh_abcxyz_h", lang))
    st.caption(t("wh_abcxyz_help", lang))
    classes = get_abc_xyz()
    matrix = classes.groupby(["abc_class", "xyz_class"]).size().unstack(fill_value=0)
    share = (
        classes.groupby("abc_class")["total_demand"].sum() / classes["total_demand"].sum() * 100
    ).round(1)
    left, right = st.columns([2, 1])
    left.dataframe(matrix, use_container_width=True)
    right.dataframe(
        share.rename("% " + ("talep payı" if tr else "of demand")).to_frame(),
        use_container_width=True,
    )


def view_optimization_map(lang: str, tables, context):
    st.subheader(t("map_h", lang))
    st.markdown(t("map_intro", lang))

    deltas = get_report("scenario_deltas.csv")
    gains: dict[str, float] = {}
    if deltas is not None:
        on_time = deltas[deltas["kpi"] == "on_time_rate"]
        gains = {
            row.scenario: 100 * row.delta_mean for row in on_time.itertuples(index=False)
        }

    st.pyplot(draw_layer_diagram(lang, gains), use_container_width=True)

    st.subheader(t("map_flow_h", lang))
    st.pyplot(draw_order_stages(lang), use_container_width=True)

    st.subheader(t("map_table_h", lang))
    tr = lang == "tr"

    def gain_text(key: str, distance_key: str | None = None) -> str:
        parts = []
        if key in gains:
            parts.append(f"{gains[key]:+.1f} pp {'zamanında' if tr else 'on-time'}")
        if deltas is not None and distance_key:
            dist = deltas[(deltas["kpi"] == "distance_per_line_m") & (deltas["scenario"] == key)]
            if not dist.empty:
                parts.append(f"{dist['relative_pct'].iloc[0]:+.1f}% {'mesafe' if tr else 'distance'}")
        return " · ".join(parts) if parts else "—"

    rows = [
        (
            "Taktik" if tr else "Tactical",
            "Haftalık" if tr else "Weekly",
            "Ürün → raf gözü" if tr else "SKU to slot",
            "ABC (geçmiş talep)" if tr else "ABC on trailing demand",
            "Tahmin × aciliyet, kısıtlı atama" if tr else "Forecast x urgency, constrained assignment",
            gain_text("slotting_only", "slotting_only"),
        ),
        (
            "Operasyonel" if tr else "Operational",
            "Dakikalar" if tr else "Minutes",
            "Sıradaki sipariş + gruplama" if tr else "Next order + batching",
            "EDD (en erken teslim)" if tr else "EDD (earliest due date)",
            "Triage (Moore-Hodgson)",
            gain_text("dispatch_only", "dispatch_only"),
        ),
        (
            "Anlık" if tr else "Immediate",
            "Saniyeler" if tr else "Seconds",
            "Durak sırası" if tr else "Stop sequence",
            "S-shape",
            "En yakın komşu + 2-opt" if tr else "Nearest neighbour + 2-opt",
            gain_text("routing_only", "routing_only"),
        ),
    ]
    st.dataframe(
        pd.DataFrame(
            rows,
            columns=[
                t("map_col_layer", lang), t("map_col_horizon", lang),
                t("map_col_decision", lang), t("map_col_baseline", lang),
                t("map_col_optimized", lang), t("map_col_gain", lang),
            ],
        ),
        use_container_width=True, hide_index=True,
    )


def view_placement(lang: str, tables, context):
    """Half one of shipping: what goes near the ramp, and on what basis."""
    tr = lang == "tr"
    st.subheader(t("two_halves_h", lang))
    st.markdown(t("two_halves", lang))
    st.divider()

    st.subheader(t("pl_h", lang))
    st.markdown(t("pl_intro", lang))

    if context is None:
        st.warning(t("artifacts_missing", lang))
        return

    catalog = tables["sku_catalog"]
    slots = tables["warehouse_slots"].set_index("slot_id")

    # ---- the score, recomputed live so beta can be explored -----------
    st.markdown(f"#### {t('pl_beta_h', lang)}")
    st.caption(t("pl_beta_help", lang))
    beta = st.slider(
        "β — " + ("aciliyet ağırlığı" if tr else "urgency weight"),
        0.0, 6.0, float(CONFIG.policy.urgency_beta), 0.5,
    )

    visits = context.forecast_visits.reindex(catalog["sku_id"], fill_value=0.0)
    share = context.urgency_share.reindex(catalog["sku_id"], fill_value=0.0)
    scored = pd.DataFrame(
        {
            "sku_id": catalog["sku_id"].to_numpy(),
            "category": catalog["category"].to_numpy(),
            "forecast_picks": visits.to_numpy(),
            "same_day_share": share.to_numpy(),
        }
    )
    scored["urgency_weight"] = 1.0 + beta * scored["same_day_share"]
    scored["score"] = scored["forecast_picks"] * scored["urgency_weight"]
    scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    assigned = context.urgency_assignment.set_index("sku_id")["slot_id"]
    scored["distance_m"] = (
        scored["sku_id"].map(assigned).map(slots["dock_distance"]).to_numpy()
    )

    # ---- explain one product -------------------------------------------
    st.markdown(f"#### {t('pl_why_h', lang)}")
    sku_id = st.selectbox("SKU", scored["sku_id"].tolist(), key="placement_sku")
    row = scored[scored["sku_id"] == sku_id].iloc[0]

    cols = st.columns(5)
    cols[0].metric(t("pl_forecast_picks", lang), f"{row['forecast_picks']:.0f}")
    cols[1].metric(t("pa_urgency", lang), f"{100 * row['same_day_share']:.0f}%")
    cols[2].metric(t("pl_urgency_w", lang), f"×{row['urgency_weight']:.2f}")
    cols[3].metric(t("pl_score", lang), f"{row['score']:.0f}")
    cols[4].metric(t("pl_rank", lang), f"{int(row['rank'])} / {len(scored)}")

    st.caption(
        (f"{row['forecast_picks']:.0f} toplama × {row['urgency_weight']:.2f} aciliyet "
         f"= {row['score']:.0f} skor → {len(scored)} ürün içinde "
         f"{int(row['rank'])}. sırada → dock'a {row['distance_m']:.1f} m"
         if tr else
         f"{row['forecast_picks']:.0f} picks x {row['urgency_weight']:.2f} urgency "
         f"= {row['score']:.0f} score -> ranked {int(row['rank'])} of {len(scored)} "
         f"-> {row['distance_m']:.1f} m from the dock")
    )

    # ---- who is near, who is far ---------------------------------------
    show = ["sku_id", "category", "forecast_picks", "same_day_share",
            "urgency_weight", "score", "distance_m"]
    headers = (
        ["SKU", "Kategori", "Tahmini toplama", "Aynı-gün payı", "Aciliyet", "Skor", "Mesafe (m)"]
        if tr else
        ["SKU", "Category", "Forecast picks", "Same-day share", "Urgency", "Score", "Distance (m)"]
    )
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{t('pl_near_h', lang)}**")
        near = scored.nsmallest(15, "distance_m")[show]
        near.columns = headers
        st.dataframe(near.round(2), use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"**{t('pl_far_h', lang)}**")
        far = scored.nlargest(15, "distance_m")[show]
        far.columns = headers
        st.dataframe(far.round(2), use_container_width=True, hide_index=True)

    # ---- profile by distance --------------------------------------------
    st.markdown(f"#### {t('pl_profile_h', lang)}")
    st.caption(t("pl_profile_help", lang))

    classes = get_abc_xyz().set_index("sku_id")
    scored["abc_class"] = scored["sku_id"].map(classes["abc_class"])
    band = 10.0
    scored["band"] = (scored["distance_m"] // band * band).astype(float)

    profile = (
        scored.dropna(subset=["band"])
        .groupby(["band", "abc_class"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    fig, ax = new_figure(8.0, 3.4)
    bottom = np.zeros(len(profile))
    colours = {"A": PALETTE["amber"], "B": PALETTE["steel"], "C": PALETTE["steel_dim"]}
    for cls in [c for c in ("A", "B", "C") if c in profile.columns]:
        ax.bar(profile.index, profile[cls], width=band * 0.85, bottom=bottom,
               label=f"{cls}", color=colours[cls])
        bottom += profile[cls].to_numpy()
    ax.set_xlabel("dock'a mesafe (m)" if tr else "distance from dock (m)")
    ax.set_ylabel("ürün sayısı" if tr else "products")
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)


def view_evacuation(lang: str, tables, context):
    """Half two of shipping: finding the stock and getting it onto the vehicle."""
    tr = lang == "tr"
    st.subheader(t("ev_h", lang))
    st.markdown(t("ev_intro", lang))

    raw = get_report("scenario_raw.csv")
    if raw is None:
        st.warning(t("no_results_yet", lang))
        return

    means = raw.groupby("scenario", sort=False).mean(numeric_only=True)

    # ---- miss attribution -----------------------------------------------
    st.markdown(f"#### {t('ev_attrib_h', lang)}")
    if "miss_share__picking" in means.columns:
        fig, ax = new_figure(8.0, 2.8)
        order = [s for s in ("baseline", "optimized") if s in means.index]
        y = np.arange(len(order))
        pick = means.loc[order, "miss_share__picking"] * 100
        load = means.loc[order, "miss_share__loading"] * 100
        ax.barh(y, pick, color=PALETTE["amber"], label=t("ev_miss_picking", lang))
        ax.barh(y, load, left=pick, color=PALETTE["blue"], label=t("ev_miss_loading", lang))
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlabel("kaçırılan siparişlerin dağılımı (%)" if tr
                      else "share of missed orders (%)")
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8, loc="lower right")
        st.pyplot(fig, use_container_width=True)

        base = means.loc["baseline"]
        card(
            t("ev_ceiling_h", lang),
            (f"Tabanda kaçırılan siparişlerin **%{100 * base['miss_share__loading']:.0f}"
             f"**'ı yükleme kaynaklı. Toplama, gruplama ve rota optimizasyonu bu kısma "
             f"dokunamaz — ne kadar iyi toplarsanız toplayın, mal hazır olduğu hâlde "
             f"ekip yetişemiyorsa kamyon onsuz kalkar. Kalan **%"
             f"{100 * base['miss_share__picking']:.0f}**'lık bölüm ise optimizasyonun "
             f"gerçekten çalışabileceği alandır."
             if tr else
             f"**{100 * base['miss_share__loading']:.0f}%** of the baseline's missed "
             f"orders are caused by loading. Picking, batching and routing cannot touch "
             f"that share — however well you pick, if the stock is ready and the crew "
             f"cannot get to it, the truck leaves without it. The remaining "
             f"**{100 * base['miss_share__picking']:.0f}%** is where optimisation can "
             f"actually work."),
            kind="negative",
        )

    # ---- headline loading metrics ---------------------------------------
    cols = st.columns(4)
    if "mean_staging_dwell_h" in means.columns:
        cols[0].metric(t("ev_staging_dwell", lang),
                       f"{means.loc['baseline', 'mean_staging_dwell_h']:.1f} h")
    if "loading_window_used_pct" in means.columns:
        cols[1].metric(t("ev_window_used", lang),
                       f"{means.loc['baseline', 'loading_window_used_pct']:.0f}%")
    if "mean_orders_per_truck" in means.columns:
        cols[2].metric(t("ev_orders_per_truck", lang),
                       f"{means.loc['baseline', 'mean_orders_per_truck']:.0f}")
    cols[3].metric(
        "Dalga penceresi" if tr else "Release window",
        f"{CONFIG.policy.release_horizon_h:.0f} h",
    )

    # ---- pipeline --------------------------------------------------------
    st.markdown(f"#### {t('ev_pipeline_h', lang)}")
    st.pyplot(draw_order_stages(lang), use_container_width=True)

    st.markdown(f"#### {t('ev_release_h', lang)}")
    st.caption(t("ev_release_help", lang))

    # ---- comparison table ------------------------------------------------
    loading_cols = [
        c for c in (
            "on_time_rate", "miss_share__picking", "miss_share__loading",
            "mean_staging_dwell_h", "p95_staging_dwell_h",
            "loading_window_used_pct", "mean_orders_per_truck",
            "rolled_picking", "rolled_loading",
        )
        if c in means.columns
    ]
    st.markdown(f"#### {t('ev_truck_h', lang)}")
    st.caption(t("ev_truck_help", lang))
    st.dataframe(means[loading_cols].round(3), use_container_width=True)


def view_journey(lang: str, tables, context):
    st.subheader(t("jr_h", lang))
    st.markdown(t("jr_intro", lang))

    if context is None:
        st.warning(t("artifacts_missing", lang))
        return

    tr = lang == "tr"
    cfg = CONFIG.warehouse
    orders = tables["orders"]
    lines = tables["order_lines"]
    slots = tables["warehouse_slots"].set_index("slot_id")

    sizes = lines.groupby("order_id").size()
    candidates = sizes[(sizes >= 6) & (sizes <= 18)].index.tolist()
    if not candidates:
        st.info(t("jr_no_orders", lang))
        return

    order_id = st.selectbox(t("jr_select_order", lang), candidates[:300])
    header = orders[orders["order_id"] == order_id].iloc[0]
    picks = lines[lines["order_id"] == order_id].reset_index(drop=True)

    assignment = context.urgency_assignment.set_index("sku_id")["slot_id"]
    picks["slot_id"] = picks["sku_id"].map(assignment)
    picks = picks.dropna(subset=["slot_id"]).reset_index(drop=True)
    stops = slots.loc[picks["slot_id"]].reset_index()
    stops["sku_id"] = picks["sku_id"].to_numpy()
    stops["units"] = picks["units"].to_numpy()

    xs, ys = stops["x"].to_numpy(), stops["y"].to_numpy()
    cross = cfg.cross_aisle_ys
    s_order, s_len = route_s_shape(xs, ys, cfg.dock_x, cross)
    n_order, n_len = route_nn_2opt(xs, ys, cfg.dock_x, cross)

    cols = st.columns(4)
    cols[0].metric(t("jr_service", lang), header["service_level"])
    cols[1].metric(t("jr_lines", lang), int(len(stops)))
    cols[2].metric("S-shape", f"{s_len:.0f} m")
    cols[3].metric(
        "NN + 2-opt", f"{n_len:.0f} m",
        f"{100 * (n_len / s_len - 1):+.1f}%", delta_color="inverse",
    )

    # ---------------- animation ----------------
    st.markdown(f"#### {t('jr_route_h', lang)}")
    st.caption(t("jr_route_help", lang))

    # A slider needs a range; a one-stop tour has none.
    n_stops = int(len(n_order))
    if n_stops > 1:
        step = st.slider("Durak" if tr else "Stop", 1, n_stops, 1)
    else:
        step = 1

    html = render_journey_svg(
        stops, n_order,
        highlight=step,
        cfg=cfg,
        labels={
            "dock": "DOCK",
            "cold": "SOĞUK BÖLGE" if tr else "COLD ZONE",
            "play": "Oynat" if tr else "Play",
            "pause": "Duraklat" if tr else "Pause",
            "speed": "Hız" if tr else "Speed",
        },
    )
    components.html(html, height=660, scrolling=False)

    # ---------------- per-stop detail ----------------
    dist = build_distance_matrix(xs, ys, cfg.dock_x, cross)
    cumulative, previous = [], 0
    running = 0.0
    for rank, idx in enumerate(n_order):
        running += dist[previous, idx + 1]
        cumulative.append(running)
        previous = idx + 1

    picking = CONFIG.picking
    detail = pd.DataFrame(
        {
            ("Sıra" if tr else "Stop"): np.arange(1, len(n_order) + 1),
            "SKU": stops["sku_id"].to_numpy()[n_order],
            ("Raf gözü" if tr else "Slot"): stops["slot_id"].to_numpy()[n_order],
            ("Koridor" if tr else "Aisle"): stops["aisle"].to_numpy()[n_order],
            ("Kat" if tr else "Level"): stops["level"].to_numpy()[n_order],
            ("Adet" if tr else "Units"): stops["units"].to_numpy()[n_order].astype(int),
            ("Kümülatif mesafe (m)" if tr else "Cumulative distance (m)"):
                np.round(cumulative, 1),
        }
    )
    detail[("Kümülatif süre (dk)" if tr else "Cumulative time (min)")] = np.round(
        (
            np.asarray(cumulative) / picking.walking_speed_mps
            + picking.per_line_time_s * np.arange(1, len(n_order) + 1)
            + picking.per_unit_time_s
            * np.cumsum(stops["units"].to_numpy()[n_order])
            + picking.setup_time_s
        )
        / 60.0,
        1,
    )

    st.markdown(f"#### {t('jr_stops_h', lang)}")
    st.dataframe(
        detail.style.apply(
            lambda row: [
                f"background-color: {PALETTE['amber_dim']}" if row.name == step - 1 else ""
                for _ in row
            ],
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )

    # ---------------- timeline ----------------
    st.markdown(f"#### {t('jr_timeline_h', lang)}")
    st.caption(t("jr_timeline_help", lang))
    st.pyplot(draw_timeline(header, lang), use_container_width=True)


def view_putaway(lang: str, tables, context):
    st.subheader(t("pa_h", lang))
    st.markdown(t("pa_intro", lang))

    if context is None:
        st.warning(t("artifacts_missing", lang))
        return

    tr = lang == "tr"
    catalog = tables["sku_catalog"]
    slots = tables["warehouse_slots"]

    sku_id = st.selectbox(t("pa_sku", lang), catalog["sku_id"].tolist())
    sku = catalog[catalog["sku_id"] == sku_id].iloc[0]

    constraints = []
    if bool(sku["is_cold"]):
        constraints.append(t("pa_cold", lang))
    if bool(sku["is_heavy"]):
        constraints.append(t("pa_heavy", lang))

    cols = st.columns(4)
    cols[0].metric(t("pa_category", lang), sku["category"])
    cols[1].metric(t("pa_forecast", lang), f"{context.forecast_visits.get(sku_id, 0):.0f}")
    cols[2].metric(t("pa_urgency", lang), f"{100 * context.urgency_share.get(sku_id, 0.0):.0f}%")
    cols[3].metric(t("pa_constraints", lang), ", ".join(constraints) or t("pa_none", lang))

    recommended = context.urgency_assignment[context.urgency_assignment["sku_id"] == sku_id]
    baseline = context.abc_assignment[context.abc_assignment["sku_id"] == sku_id]
    if recommended.empty or baseline.empty:
        st.info(t("jr_no_orders", lang))
        return

    rec = slots[slots["slot_id"] == recommended["slot_id"].iloc[0]].iloc[0]
    base = slots[slots["slot_id"] == baseline["slot_id"].iloc[0]].iloc[0]
    delta = rec["dock_distance"] - base["dock_distance"]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"**{t('pa_recommended', lang)}** `{rec['slot_id']}`  \n"
            f"{'koridor' if tr else 'aisle'} {rec['aisle']}, "
            f"{'göz' if tr else 'bay'} {rec['bay']}, "
            f"{'kat' if tr else 'level'} {rec['level']}  \n"
            f"{t('pa_walk', lang)}: **{rec['dock_distance']:.1f} m**"
        )
        st.markdown(
            f"**{t('pa_baseline_slot', lang)}** `{base['slot_id']}` "
            f"({base['dock_distance']:.1f} m) — {t('pa_diff', lang)}: **{delta:+.1f} m**"
        )
    with right:
        fig, ax = new_figure(5.4, 4.4)
        draw_floor_plan(ax, CONFIG.warehouse, show_labels=False)
        ax.scatter([base["x"]], [base["y"]], s=190, marker="s", facecolors="none",
                   edgecolors=PALETTE["steel"], linewidths=2,
                   label="Taban" if tr else "Baseline", zorder=6)
        ax.scatter([rec["x"]], [rec["y"]], s=190, marker="s", facecolors="none",
                   edgecolors=PALETTE["red"], linewidths=2.5,
                   label="Önerilen" if tr else "Recommended", zorder=6)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.2)
        st.pyplot(fig, use_container_width=True)


def view_results(lang: str, tables, context):
    st.subheader(t("rs_h", lang))
    tr = lang == "tr"

    raw = get_report("scenario_raw.csv")
    deltas = get_report("scenario_deltas.csv")

    if raw is None:
        st.warning(t("no_results_yet", lang))
        return

    headline = [
        "on_time_rate", "distance_per_line_m", "mean_tardiness_h",
        "p95_tardiness_h", "mean_orders_per_tour", "picker_utilisation",
    ]
    available = [c for c in headline if c in raw.columns]
    summary = raw.groupby("scenario", sort=False)[available].mean()
    summary.insert(0, "missed_pct", 100 * (1 - summary["on_time_rate"]))

    best = summary["missed_pct"].idxmin()
    base_missed = summary.loc["baseline", "missed_pct"]

    cols = st.columns(3)
    cols[0].metric(f"{t('rs_missed', lang)} — {t('rs_baseline', lang)}", f"{base_missed:.2f}%")
    cols[1].metric(
        f"{t('rs_best', lang)} ({best})", f"{summary.loc[best, 'missed_pct']:.2f}%",
        f"{100 * (summary.loc[best, 'missed_pct'] / base_missed - 1):.0f}%",
        delta_color="inverse",
    )
    cols[2].metric(
        t("rs_distance", lang), f"{summary.loc[best, 'distance_per_line_m']:.2f} m",
        f"{100 * (summary.loc[best, 'distance_per_line_m'] / summary.loc['baseline', 'distance_per_line_m'] - 1):.1f}%",
        delta_color="inverse",
    )

    st.markdown(f"#### {t('rs_table_h', lang)}")
    st.caption(t("rs_table_help", lang))
    st.dataframe(summary.round(4), use_container_width=True)

    if deltas is not None:
        st.markdown(f"#### {t('rs_delta_h', lang)}")
        st.caption(t("rs_delta_help", lang))
        for kpi in ("on_time_rate", "distance_per_line_m"):
            sub = deltas[deltas["kpi"] == kpi]
            if sub.empty:
                continue
            fig, ax = new_figure(8.0, 2.9)
            colours = [PALETTE["green"] if v > 0 else PALETTE["red"] for v in sub["delta_mean"]]
            if kpi.startswith("distance"):
                colours = [PALETTE["green"] if v < 0 else PALETTE["red"] for v in sub["delta_mean"]]
            ax.barh(
                sub["scenario"], sub["delta_mean"],
                xerr=[sub["delta_mean"] - sub["delta_ci95_low"],
                      sub["delta_ci95_high"] - sub["delta_mean"]],
                color=colours, capsize=4,
            )
            ax.axvline(0, color=PALETTE["steel"], lw=1)
            ax.set_xlabel(f"Δ {kpi} (95% CI)")
            st.pyplot(fig, use_container_width=True)

    # ---------------- capacity ----------------
    curve = get_report("load_curve_raw.csv")
    if curve is not None:
        st.divider()
        st.markdown(f"#### {t('rs_capacity_h', lang)}")
        st.caption(t("rs_capacity_help", lang))
        pivot = curve.pivot_table(index="orders_per_day", columns="scenario",
                                  values="on_time_rate")
        caps = {}
        for scenario in ("baseline", "optimized"):
            ok = pivot.index[pivot[scenario] >= 0.95]
            caps[scenario] = ok.max() if len(ok) else None

        fig, ax = new_figure(8.0, 3.6)
        ax.plot(pivot.index, pivot["baseline"], "o--", lw=2, ms=7,
                color=PALETTE["steel"], label="Taban" if tr else "Baseline")
        ax.plot(pivot.index, pivot["optimized"], "s-", lw=2, ms=7,
                color=PALETTE["amber"], label="Optimize" if tr else "Optimised")
        ax.axhline(0.95, color=PALETTE["red"], ls=":", lw=1.4)
        ax.set_xlabel(t("orders_per_day", lang))
        ax.set_ylabel(t("on_time_rate", lang))
        ax.legend(fontsize=8)
        if all(caps.values()):
            ax.annotate("", xy=(caps["optimized"], 0.95), xytext=(caps["baseline"], 0.95),
                        arrowprops=dict(arrowstyle="<->", color=PALETTE["red"], lw=2))
            ax.annotate(
                f"+{100 * (caps['optimized'] / caps['baseline'] - 1):.0f}% "
                + ("kapasite" if tr else "throughput"),
                ((caps["baseline"] + caps["optimized"]) / 2, 0.88),
                ha="center", color=PALETTE["red"], fontsize=11, weight="bold",
            )
        st.pyplot(fig, use_container_width=True)

    # ---------------- models ----------------
    st.divider()
    st.markdown(f"#### {t('rs_models_h', lang)}")
    st.caption(t("rs_models_help", lang))
    for title, name in (
        ("ML-1 " + ("talep tahmini" if tr else "demand forecast"), "model_demand_backtest.csv"),
        ("ML-2 " + ("toplama süresi" if tr else "pick time"), "model_pick_time.csv"),
        ("ML-3 " + ("SLA riski" if tr else "SLA risk"), "model_sla_risk.csv"),
    ):
        report = get_report(name)
        if report is not None:
            st.markdown(f"*{title}*")
            st.dataframe(report.round(4), use_container_width=True, hide_index=True)

    dispatch = get_report("dispatch_study.csv")
    if dispatch is not None:
        st.markdown(f"*{'Sevkiyat kuralı karşılaştırması' if tr else 'Dispatch rule study'}*")
        st.dataframe(dispatch.round(4), use_container_width=True)

    stress = get_report("stress_summary.csv")
    if stress is not None:
        st.divider()
        st.markdown(f"#### {t('rs_stress_h', lang)}")
        st.caption(t("rs_stress_help", lang))
        st.dataframe(stress.round(4), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------


def main():
    apply_matplotlib_style()
    st.markdown(CSS, unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.session_state["lang"] = "tr"

    with st.sidebar:
        st.markdown("### 📦 " + t("app_title", st.session_state["lang"]))
        # `key="lang"` binds the widget to session_state, which already holds the
        # Turkish default set above; passing `index` as well would fight it.
        lang = st.radio(
            t("language", st.session_state["lang"]),
            options=list(LANGUAGES.keys()),
            format_func=lambda code: LANGUAGES[code],
            horizontal=True,
            key="lang",
        )

        tables = get_tables()
        context = get_context()
        cfg = CONFIG.warehouse

        st.divider()
        st.markdown(f"**{t('sidebar_facility', lang)}**")
        st.metric(t("slots", lang), f"{len(tables['warehouse_slots']):,}")
        st.metric(t("skus", lang), f"{len(tables['sku_catalog']):,}")
        st.metric(t("orders_window", lang), f"{len(tables['orders']):,}")
        st.metric(
            t("occupancy", lang),
            f"{100 * len(tables['sku_catalog']) / len(tables['warehouse_slots']):.0f}%",
        )
        st.caption(
            f"{cfg.n_aisles} × {cfg.bays_per_aisle} × {cfg.levels} · "
            f"{cfg.n_cross_aisles} "
            + ("çapraz koridor" if lang == "tr" else "cross aisles")
        )
        meta = tables["_meta"].iloc[0]
        st.caption(f"{t('calibration_source', lang)}: **{meta['calibration_source']}**")
        if context is None:
            st.error(t("artifacts_missing", lang))

    st.title(t("app_title", lang))
    st.caption(t("app_subtitle", lang))

    tabs = st.tabs([
        t("tab_overview", lang),
        t("tab_warehouse", lang),
        t("tab_placement", lang),
        t("tab_evacuation", lang),
        t("tab_journey", lang),
        t("tab_map", lang),
        t("tab_putaway", lang),
        t("tab_results", lang),
    ])
    views = (
        view_overview, view_warehouse, view_placement, view_evacuation,
        view_journey, view_optimization_map, view_putaway, view_results,
    )
    for tab, view in zip(tabs, views):
        with tab:
            view(lang, tables, context)


if __name__ == "__main__":
    main()
