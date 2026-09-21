"""Animated floor plan for the order journey, emitted as self-contained SVG.

Why SVG rather than a sequence of matplotlib frames driven by Streamlit reruns:
a rerun-driven animation re-lays out the whole page on every tick, which is the
same layout thrash that made this dashboard unusable in the first place. An SVG
with ``<animateMotion>`` plays in the browser, touches nothing else on the page,
and keeps working when the server is busy.

The picker marker follows the *actual* route the routing policy produced, drawn
under the aisle travel model -- it walks out of an aisle, along a cross aisle and
back in, never through a rack. Play, pause and speed are plain inline JS with no
external dependencies, so this renders under Streamlit's CSP without a CDN.
"""

from __future__ import annotations

import html

import numpy as np

from config import CONFIG, WarehouseConfig
from app.theme import PALETTE

__all__ = ["build_route_polyline", "render_journey_svg"]


def _nearest_cross_aisle(y: float, cross: tuple[float, ...]) -> float:
    return min(cross, key=lambda c: abs(c - y))


def build_route_polyline(
    xs: np.ndarray,
    ys: np.ndarray,
    order: np.ndarray,
    cfg: WarehouseConfig | None = None,
    *,
    with_stop_indices: bool = False,
):
    """Expand a visiting order into the path a picker physically walks.

    A straight line between two stops in different aisles would cut through the
    racking. The real path leaves the aisle at whichever cross aisle is cheapest,
    crosses, and re-enters -- precisely the assumption the distance model is
    built on, so the drawing and the measured distance agree.

    With ``with_stop_indices`` the positions of the actual pick locations within
    the returned path are also given, which is what lets each stop marker light
    up at the moment the picker truly arrives rather than on a guessed schedule.
    """
    cfg = cfg or CONFIG.warehouse
    cross = cfg.cross_aisle_ys

    path: list[tuple[float, float]] = [(cfg.dock_x, 0.0)]
    stop_indices: list[int] = []
    current = (cfg.dock_x, 0.0)

    for position, idx in enumerate(list(order) + [None]):
        if idx is None:
            target = (cfg.dock_x, 0.0)
        else:
            target = (float(xs[idx]), float(ys[idx]))

        if abs(target[0] - current[0]) < 1e-9:
            path.append(target)
        else:
            # Choose the cross aisle that minimises the detour, exactly as the
            # distance matrix does.
            best = min(cross, key=lambda c: abs(current[1] - c) + abs(target[1] - c))
            path.append((current[0], best))
            path.append((target[0], best))
            path.append(target)

        if idx is not None:
            stop_indices.append(len(path) - 1)
        current = target

    if with_stop_indices:
        return path, stop_indices
    return path


def _polyline_length(points: list[tuple[float, float]], to_px) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        ax, ay = to_px(*a)
        bx, by = to_px(*b)
        total += float(np.hypot(bx - ax, by - ay))
    return total


def _arrival_fractions(
    points: list[tuple[float, float]], stop_indices: list[int], to_px
) -> list[float]:
    """Fraction of the total path length at which each stop is reached."""
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        ax, ay = to_px(*a)
        bx, by = to_px(*b)
        cumulative.append(cumulative[-1] + float(np.hypot(bx - ax, by - ay)))

    total = cumulative[-1] or 1.0
    return [cumulative[i] / total for i in stop_indices]


def _svg_path(points: list[tuple[float, float]], to_px) -> str:
    commands = []
    for i, (x, y) in enumerate(points):
        px, py = to_px(x, y)
        commands.append(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}")
    return " ".join(commands)


def render_journey_svg(
    stops: "object",
    order: np.ndarray,
    *,
    highlight: int | None = None,
    cfg: WarehouseConfig | None = None,
    width: int = 900,
    duration_s: float = 12.0,
    labels: dict[str, str] | None = None,
) -> str:
    """Return a complete HTML document drawing and animating one picking tour.

    ``stops`` is a frame with ``x``, ``y`` and ``slot_id``; ``order`` is the
    visiting sequence produced by a routing policy.
    """
    cfg = cfg or CONFIG.warehouse
    labels = labels or {}

    xs = stops["x"].to_numpy(dtype=float)
    ys = stops["y"].to_numpy(dtype=float)

    # --- world -> pixel ------------------------------------------------
    pad = 46
    world_w = (cfg.n_aisles - 1) * cfg.aisle_width + cfg.aisle_width * 2
    world_h = cfg.aisle_length + 10
    scale = (width - 2 * pad) / world_w
    height = int(world_h * scale + 2 * pad)

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = pad + (x + cfg.aisle_width) * scale
        py = height - pad - (y + 5.0) * scale
        return px, py

    parts: list[str] = []

    # --- cold zone -----------------------------------------------------
    if cfg.cold_zone_aisles:
        cold = sorted(cfg.cold_zone_aisles)
        x0, _ = to_px(cold[0] * cfg.aisle_width - cfg.aisle_width * 0.55, 0)
        x1, _ = to_px(cold[-1] * cfg.aisle_width + cfg.aisle_width * 0.55, 0)
        _, y_top = to_px(0, cfg.aisle_length + 1.5)
        _, y_bot = to_px(0, -3.0)
        parts.append(
            f'<rect x="{x0:.1f}" y="{y_top:.1f}" width="{x1 - x0:.1f}" '
            f'height="{y_bot - y_top:.1f}" fill="{PALETTE["cold"]}" opacity="0.10" '
            f'stroke="{PALETTE["cold"]}" stroke-dasharray="4 3" stroke-width="1"/>'
            f'<text x="{(x0 + x1) / 2:.1f}" y="{y_top - 6:.1f}" fill="{PALETTE["cold"]}" '
            f'font-size="10" font-family="monospace" text-anchor="middle">'
            f'{html.escape(labels.get("cold", "COLD ZONE"))}</text>'
        )

    # --- rack blocks ---------------------------------------------------
    rack_w = cfg.aisle_width * 0.36 * scale
    bay_h = cfg.bay_depth * 0.92 * scale
    offset = cfg.aisle_width * 0.30
    for aisle in range(cfg.n_aisles):
        x_centre = aisle * cfg.aisle_width
        for sign in (-1, 1):
            gx, _ = to_px(x_centre + sign * offset, 0)
            for bay in range(cfg.bays_per_aisle):
                y_world = cfg.cross_aisle_offset + bay * cfg.bay_depth
                _, gy = to_px(0, y_world)
                parts.append(
                    f'<rect x="{gx - rack_w / 2:.1f}" y="{gy - bay_h / 2:.1f}" '
                    f'width="{rack_w:.1f}" height="{bay_h:.1f}" '
                    f'fill="{PALETTE["steel_dim"]}" opacity="0.45"/>'
                )

    # --- cross aisles ---------------------------------------------------
    for i, y in enumerate(cfg.cross_aisle_ys):
        x0, py = to_px(-cfg.aisle_width, y)
        x1, _ = to_px((cfg.n_aisles - 1) * cfg.aisle_width + cfg.aisle_width, y)
        name = "FRONT" if i == 0 else ("BACK" if i == len(cfg.cross_aisle_ys) - 1 else "MID")
        parts.append(
            f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{py:.1f}" '
            f'stroke="{PALETTE["amber"]}" stroke-width="1" stroke-dasharray="6 4" opacity="0.5"/>'
            f'<text x="{x1 + 4:.1f}" y="{py + 3:.1f}" fill="{PALETTE["amber"]}" '
            f'font-size="9" font-family="monospace" opacity="0.8">{name}</text>'
        )

    # --- dock -----------------------------------------------------------
    dx, dy = to_px(cfg.dock_x, 0.0)
    parts.append(
        f'<rect x="{dx - 22:.1f}" y="{dy + 6:.1f}" width="44" height="16" rx="2" '
        f'fill="{PALETTE["blue"]}"/>'
        f'<text x="{dx:.1f}" y="{dy + 18:.1f}" fill="#04121f" font-size="10" '
        f'font-family="monospace" font-weight="700" text-anchor="middle">'
        f'{html.escape(labels.get("dock", "DOCK"))}</text>'
    )

    # --- the route -------------------------------------------------------
    polyline, stop_indices = build_route_polyline(
        xs, ys, order, cfg, with_stop_indices=True
    )
    path_d = _svg_path(polyline, to_px)
    parts.append(
        f'<path id="route" d="{path_d}" fill="none" stroke="{PALETTE["steel_dim"]}" '
        f'stroke-width="2" stroke-linejoin="round" opacity="0.7"/>'
    )
    # A second copy of the same path, drawn on top and revealed by animating its
    # dash offset. The picker therefore leaves a completed trail behind it.
    total_len = _polyline_length(polyline, to_px)
    parts.append(
        f'<path id="progress" d="{path_d}" fill="none" stroke="{PALETTE["green"]}" '
        f'stroke-width="2.5" stroke-linejoin="round" opacity="0.9" '
        f'stroke-dasharray="{total_len:.1f}" stroke-dashoffset="{total_len:.1f}">'
        f'<animate id="reveal" attributeName="stroke-dashoffset" '
        f'from="{total_len:.1f}" to="0" dur="{duration_s:.1f}s" '
        f'repeatCount="indefinite" begin="0s"/>'
        f"</path>"
    )

    # --- stop markers ----------------------------------------------------
    # Each stop starts amber and flips to green at the moment the picker
    # reaches it, then resets when the animation loops. The timing comes from
    # how far along the route the stop actually sits, so the colour change and
    # the marker arrive together rather than merely looking plausible.
    arrivals = _arrival_fractions(polyline, stop_indices, to_px)

    for rank, idx in enumerate(order, start=1):
        sx, sy = to_px(float(xs[idx]), float(ys[idx]))
        is_hit = highlight is not None and rank == highlight
        radius = 11 if is_hit else 8
        at = arrivals[rank - 1] * duration_s
        marker_id = f"stop{rank}"

        # Selected-by-slider stops stay red so the table and the map agree.
        idle_fill = PALETTE["red"] if is_hit else PALETTE["panel"]
        idle_stroke = PALETTE["red"] if is_hit else PALETTE["amber"]

        parts.append(
            f'<circle id="{marker_id}" cx="{sx:.1f}" cy="{sy:.1f}" r="{radius}" '
            f'fill="{idle_fill}" stroke="{idle_stroke}" stroke-width="2">'
            f'<animate attributeName="fill" values="{idle_fill};{PALETTE["green"]}" '
            f'dur="0.25s" begin="{at:.2f}s" repeatCount="indefinite" fill="freeze"/>'
            f'<animate attributeName="stroke" values="{idle_stroke};{PALETTE["green"]}" '
            f'dur="0.25s" begin="{at:.2f}s" repeatCount="indefinite" fill="freeze"/>'
            f'<animate attributeName="r" values="{radius};{radius + 4};{radius}" '
            f'dur="0.45s" begin="{at:.2f}s" repeatCount="indefinite"/>'
            f"</circle>"
            f'<text x="{sx:.1f}" y="{sy + 3.5:.1f}" fill="{PALETTE["text"]}" '
            f'font-size="9" font-family="monospace" font-weight="700" '
            f'text-anchor="middle" pointer-events="none">{rank}</text>'
        )

    # --- the moving picker -----------------------------------------------
    parts.append(
        f"""
  <g id="picker">
    <circle r="9" fill="{PALETTE['green']}" opacity="0.25"/>
    <circle r="5" fill="{PALETTE['green']}"/>
    <animateMotion id="walk" dur="{duration_s:.1f}s" repeatCount="indefinite"
                   rotate="auto" begin="0s">
      <mpath href="#route"/>
    </animateMotion>
  </g>"""
    )

    svg = (
        f'<svg id="floor" viewBox="0 0 {width} {height}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'style="max-width:100%;height:auto;display:block">' + "".join(parts) + "</svg>"
    )

    play_label = html.escape(labels.get("play", "Oynat"))
    pause_label = html.escape(labels.get("pause", "Duraklat"))
    speed_label = html.escape(labels.get("speed", "Hız"))

    return f"""
<div style="background:{PALETTE['panel']};border:1px solid {PALETTE['grid']};
            border-radius:6px;padding:10px;overflow:hidden;max-width:100%">
  {svg}
  <div style="display:flex;gap:10px;align-items:center;margin-top:8px;
              font-family:monospace;font-size:12px;color:{PALETTE['steel']};
              flex-wrap:wrap">
    <button id="toggle" style="background:{PALETTE['amber']};border:none;color:#1a1206;
            padding:5px 14px;border-radius:3px;cursor:pointer;font-family:monospace;
            font-weight:700">{pause_label}</button>
    <label>{speed_label}
      <input id="speed" type="range" min="0.25" max="3" step="0.25" value="1"
             style="vertical-align:middle;width:120px">
    </label>
    <span id="speedval">1.00x</span>
  </div>
</div>
<script>
  (function () {{
    const svg = document.getElementById('floor');
    const toggle = document.getElementById('toggle');
    const speed = document.getElementById('speed');
    const readout = document.getElementById('speedval');
    let playing = true;

    // SMIL has no playback-rate control, so speed is applied by rescaling every
    // timing attribute. The originals are captured once and every later change
    // is computed from those -- rescaling already-rescaled values would drift
    // the stop highlights out of step with the picker.
    const timed = [...svg.querySelectorAll('animate, animateMotion')].map(function (el) {{
      return {{
        el: el,
        dur: parseFloat(el.getAttribute('dur')) || 0,
        begin: parseFloat(el.getAttribute('begin')) || 0
      }};
    }});

    toggle.addEventListener('click', function () {{
      if (playing) {{ svg.pauseAnimations(); toggle.textContent = '{play_label}'; }}
      else {{ svg.unpauseAnimations(); toggle.textContent = '{pause_label}'; }}
      playing = !playing;
    }});

    speed.addEventListener('input', function () {{
      const factor = parseFloat(speed.value);
      readout.textContent = factor.toFixed(2) + 'x';
      timed.forEach(function (t) {{
        if (t.dur) t.el.setAttribute('dur', (t.dur / factor).toFixed(3) + 's');
        t.el.setAttribute('begin', (t.begin / factor).toFixed(3) + 's');
      }});
      // Restart the timeline so the walk, the trail and every stop marker
      // share one clock again.
      svg.setCurrentTime(0);
    }});
  }})();
</script>
"""
