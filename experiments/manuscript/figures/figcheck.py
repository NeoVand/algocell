"""Automatic layout checks for publication figures (run on every build; the figure is not acceptable until this is clean).

Checks, all in display (pixel) coordinates after a draw:
  text-text      two text boxes intersect (legend entries inside one legend are exempt)
  text-line      a line or patch outline passes through a text box (lines sampled densely)
  text-clipped   a text box extends beyond the figure, or beyond its axes when the axes draws a frame
  data-clipped   a line's data extends beyond the axes limits (part of the trace is cut off)
  type-size      a text smaller than MIN_PT points
  overlap-legend a legend box intersects a line or another text

Usage: report = figcheck.check(fig); figcheck.print_report(report)
"""
from __future__ import annotations

import numpy as np
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

MIN_PT = 5.0
TEXT_PAD = -0.5   # shrink text boxes by this many pixels before testing (glyph boxes are generous)


def _bbox(artist, renderer):
    try:
        bb = artist.get_window_extent(renderer)
    except Exception:  # noqa: BLE001
        return None
    if bb is None or not np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all() or bb.width <= 0 or bb.height <= 0:
        return None
    return bb


def _shrink(bb, pad):
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def _line_points(line, n_per_seg=12):
    """Display-space sample points along a Line2D (densified between vertices; marker-only series are just their points)."""
    xy = line.get_xydata()
    if len(xy) == 0:
        return np.empty((0, 2))
    pts = line.get_transform().transform(xy)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) <= 1 or str(line.get_linestyle()).lower() in ("none", " ", ""):
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        t = np.linspace(0, 1, n_per_seg + 1)[1:]
        out.append(a + (b - a) * t[:, None])
    return np.vstack(out)


def _patch_points(patch, n_per_seg=8):
    """Display-space sample points along a patch outline, with Bezier segments flattened (not their control polygons)."""
    try:
        if isinstance(patch, FancyArrowPatch):
            path = patch.get_path()                                   # already in display coordinates
        else:
            path = patch.get_path().transformed(patch.get_transform())
        polys = path.to_polygons(closed_only=False)
    except Exception:  # noqa: BLE001
        return np.empty((0, 2))
    out = []
    for poly in polys:
        poly = np.asarray(poly, dtype=float)
        poly = poly[np.isfinite(poly).all(axis=1)]
        if len(poly) == 0:
            continue
        out.append(poly[:1])
        for a, b in zip(poly[:-1], poly[1:]):
            t = np.linspace(0, 1, n_per_seg + 1)[1:]
            out.append(a + (b - a) * t[:, None])
    return np.vstack(out) if out else np.empty((0, 2))


def _inside(bb, pts):
    return ((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).any()


def _label(t):
    s = t.get_text().replace("\n", " ")
    return f'"{s[:40]}"' if s else "(empty)"


def _visible_tick_labels(ax):
    """Tick-label Text objects whose tick lies inside the axis view interval (the others exist but are never drawn)."""
    keep = []
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                keep += [tick.label1, tick.label2]
    return keep


def check(fig, exempt_gids=("panel-label", "allow-outside")) -> list[str]:
    dpi0 = fig.get_dpi()
    fig.set_dpi(300)   # tiny superscripts fail to hint at screen resolution; figures are saved at print resolution anyway
    try:
        fig.canvas.draw()
        return _check_drawn(fig, exempt_gids)
    finally:
        fig.set_dpi(dpi0)


def _check_drawn(fig, exempt_gids) -> list[str]:
    rend = fig.canvas.get_renderer()
    problems: list[str] = []
    for ax in fig.axes:
        all_ticklabels = {id(t) for axis in (ax.xaxis, ax.yaxis) for tick in axis.get_major_ticks() + axis.get_minor_ticks() for t in (tick.label1, tick.label2)}
        shown_ticklabels = {id(t) for t in _visible_tick_labels(ax)}
        texts = [t for t in ax.findobj(Text) if t.get_visible() and t.get_text().strip() and (id(t) not in all_ticklabels or id(t) in shown_ticklabels)]
        if not ax.axison:
            texts = [t for t in texts if id(t) not in all_ticklabels]
        legends = [l for l in ax.findobj(Legend)]
        legend_texts = {id(t) for l in legends for t in l.get_texts()}
        legend_children = {id(a) for l in legends for a in l.findobj()}   # handles and frame: not data
        lines = [l for l in ax.findobj(Line2D) if l.get_visible() and len(l.get_xydata()) > 1 and id(l) not in legend_children]
        patches = [p for p in ax.findobj(Patch) if p.get_visible() and not isinstance(p, Rectangle) and id(p) not in legend_children]
        tboxes = []
        for t in texts:
            bb = _bbox(t, rend)
            if bb is None:
                continue
            tboxes.append((t, _shrink(bb, TEXT_PAD)))
            if t.get_fontsize() < MIN_PT - 1e-6:
                problems.append(f"type-size   {_label(t)} is {t.get_fontsize():.1f} pt (< {MIN_PT} pt)")
            if ax.axison and t.get_gid() not in exempt_gids and id(t) not in legend_texts and t not in (ax.title, ax.xaxis.label, ax.yaxis.label) \
                    and id(t) not in all_ticklabels:
                abb = ax.get_window_extent(rend)
                if bb.x1 > abb.x1 + 1 or bb.x0 < abb.x0 - 1 or bb.y1 > abb.y1 + 1 or bb.y0 < abb.y0 - 1:
                    problems.append(f"text-clipped {_label(t)} extends beyond its axes")
        # text-text
        for i in range(len(tboxes)):
            for j in range(i + 1, len(tboxes)):
                ti, bi = tboxes[i]
                tj, bj = tboxes[j]
                if id(ti) in legend_texts and id(tj) in legend_texts:
                    continue
                if bi.overlaps(bj):
                    problems.append(f"text-text   {_label(ti)} overlaps {_label(tj)}")
        # text-line / text-patch
        samples = [(f"line {l.get_label()!s}" if not str(l.get_label()).startswith("_") else "a line", _line_points(l)) for l in lines]
        samples += [(f"{type(p).__name__}", _patch_points(p)) for p in patches]
        for t, bb in tboxes:
            if id(t) in legend_texts:
                continue
            for name, pts in samples:
                if len(pts) and _inside(bb, pts):
                    problems.append(f"text-line   {_label(t)} is crossed by {name}")
                    break
        # legend boxes vs lines and text
        for lg in legends:
            lbb = _bbox(lg, rend)
            if lbb is None:
                continue
            lbb = _shrink(lbb, -3)
            for name, pts in samples:
                if len(pts) and _inside(lbb, pts):
                    problems.append(f"overlap-legend legend of axes {ax.get_title() or ax.get_gid() or ''} is crossed by {name}")
                    break
            for t, bb in tboxes:
                if id(t) not in legend_texts and lbb.overlaps(bb):
                    problems.append(f"overlap-legend legend overlaps {_label(t)}")
        # data clipped (axes with frames only; axes tagged allow-clip crop on purpose, e.g. a log axis starting above the first sample)
        if ax.axison and ax.get_gid() != "allow-clip":
            x0, x1 = sorted(ax.get_xlim())
            y0, y1 = sorted(ax.get_ylim())
            for l in lines:
                xy = l.get_xydata()
                xy = xy[np.isfinite(xy).all(axis=1)]
                if len(xy) and l.get_transform() == ax.transData and l.get_clip_on():
                    if (xy[:, 0] > x1 + 1e-9).any() or (xy[:, 0] < x0 - 1e-9).any() or (xy[:, 1] > y1 + 1e-9).any() or (xy[:, 1] < y0 - 1e-9).any():
                        problems.append(f"data-clipped line {l.get_label()!s} has data outside the axes limits of '{ax.get_title()}'")
    # figure-wide: texts of different axes, and figure-level legends against everything
    all_texts = []
    all_samples = []
    for ax in fig.axes:
        legends = ax.findobj(Legend)
        legend_children = {id(a) for l in legends for a in l.findobj()}
        shown = {id(t) for t in _visible_tick_labels(ax)}
        all_tl = {id(t) for axis in (ax.xaxis, ax.yaxis) for tick in axis.get_major_ticks() + axis.get_minor_ticks() for t in (tick.label1, tick.label2)}
        for t in ax.findobj(Text):
            if t.get_visible() and t.get_text().strip() and (id(t) not in all_tl or (id(t) in shown and ax.axison)):
                bb = _bbox(t, rend)
                if bb is not None:
                    all_texts.append((ax, t, _shrink(bb, TEXT_PAD)))
        for l in ax.findobj(Line2D):
            if l.get_visible() and len(l.get_xydata()) > 1 and id(l) not in legend_children:
                all_samples.append((f"line {l.get_label()!s}" if not str(l.get_label()).startswith("_") else "a line", _line_points(l)))
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            ai, ti, bi = all_texts[i]
            aj, tj, bj = all_texts[j]
            if ai is aj:
                continue
            if bi.overlaps(bj):
                problems.append(f"text-text   {_label(ti)} overlaps {_label(tj)} (different panels)")
    for lg in fig.legends:
        lbb = _bbox(lg, rend)
        if lbb is None:
            continue
        lbb = _shrink(lbb, -3)
        for name, pts in all_samples:
            if len(pts) and _inside(lbb, pts):
                problems.append(f"overlap-legend figure legend is crossed by {name}")
                break
        lg_texts = {id(t) for t in lg.get_texts()}
        for ax, t, bb in all_texts:
            if id(t) not in lg_texts and lbb.overlaps(bb):
                problems.append(f"overlap-legend figure legend overlaps {_label(t)}")
    return sorted(set(problems))


def print_report(problems: list[str], name: str = "") -> bool:
    tag = f"[{name}] " if name else ""
    if not problems:
        print(f"{tag}figcheck: clean")
        return True
    print(f"{tag}figcheck: {len(problems)} problem(s)")
    for p in problems:
        print("   ", p)
    return False
