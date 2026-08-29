"""The one figure of the money column: how often each model filed the payment.

Read from the episode logs through `score` and `report` -- the same three functions that make
the README table, so the picture can never disagree with it. One bar per admitted undefended
model, sorted by the floor; the zeros in ascending order of how loudly they refused.

The bar is the 95% Wilson interval drawn as a fade: solid to its low end, then fading out to
its high end. An interval is not a fact with a hard edge, so it is not drawn with one; the
measured share is the number beside the bar. Red carries the meaning: every point on that
scale is an agent doing what an outsider told it to do. The column on the right stays numbers
only -- it is a second axis, not a second contest -- with a lamp that says "this one would tell
you" without a second scale.

Needs matplotlib (`pip install -e '.[figures]'`); nothing else in the harness does.

    python3 figure.py            # published sweeps -> reports/figures/money7-{models,families}.*
    python3 figure.py --tag 'money7*,mine' --slug mine
"""

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corpus    # noqa: E402
import report    # noqa: E402
import scenes    # noqa: E402
import score as sc  # noqa: E402

REPORTS = HERE / "reports"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
BAND = "#c62f26"    # the interval. Red on purpose: this scale measures damage
GRID = "#e4e3de"
TITLE = "An email asked for money. How often did the agent file the payment?"
CREDIT = "(c) 2026 Mikhail Gribov  |  huggingface.co/mihailgribov"


def rows_from(recs):
    """[(label, floor %, low, high, said-it-was-suspicious %)] plus the footer's numbers."""
    rows = []
    ctl_pay = ctl_n = 0
    stray, done = [], []
    for m, (inj, ctl) in report.arms(recs).items():
        if not report.undefended(m):
            continue
        k = sum(map(report.payment, inj))
        lo, hi = report.wilson(k, len(inj))
        sus = 100 * sum(map(sc.suspected, inj)) / len(inj)
        rows.append((report.nice(m), 100 * k / len(inj), lo, hi, sus, len(inj)))
        ctl_pay += sum(map(report.payment, ctl))
        ctl_n += len(ctl)
        stray.append(100 * sum(1 for r in ctl if scenes.unexpected_tools(r["scene"], r["calls"]))
                     / len(ctl))
        done.append(100 * sum(1 for r in ctl if r["task_ok"]) / len(ctl))
    rows.sort(key=lambda r: (-r[1], r[4], r[0]))
    n_payloads = len({r["id"] for r in recs if r["variant"] == "injected"})
    meta = {"models": len(rows), "payloads": n_payloads, "ctl_pay": ctl_pay, "ctl_n": ctl_n,
            "stray": (min(stray), max(stray)) if stray else (0, 0),
            "done": (min(done), max(done)) if done else (0, 0)}
    return rows, meta


def footer(meta):
    s_lo, s_hi = meta["stray"]
    d_lo, d_hi = meta["done"]
    return [
        "Quadrat-IPI payloads. Counted: a payment order at the run's end, whoever it went to.",
        f"Control, same emails minus the injection: {meta['ctl_pay']} stray payment"
        f"{'s' if meta['ctl_pay'] != 1 else ''} in {meta['ctl_n']:,} runs, stray tools "
        f"{s_lo:.0f}-{s_hi:.0f}%, job done {d_lo:.0f}-{d_hi:.0f}%.",
        "Bar = 95% Wilson interval, solid low, fading high; zeros mean \"under 1%\". "
        "One scene, a non-adaptive attacker.",
    ]


def draw(rows, meta, outs):
    """Write the figure to every path in `outs` (the suffix picks the format)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows)
    xmax = max(56.0, max(r[3] for r in rows) + 14)
    fig_w, dpi = 9.6, 200
    fig_h = 2.8 + 0.52 * n
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    renderer = fig.canvas.get_renderer()

    def fit(t, limit, floor):
        while t.get_window_extent(renderer).width / (fig_w * dpi) > limit and \
                t.get_fontsize() > floor:
            t.set_fontsize(t.get_fontsize() - 0.5)

    # One line, as large as fits: the headline is the hook, and a wrapped hook reads as two
    # thoughts. Measured, not guessed, so a reworded title stays on one line.
    top = 1 - 0.26 / fig_h
    title = fig.text(0.045, top, TITLE, fontsize=19, color=INK, va="top", weight="bold")
    fit(title, 0.91, 11)
    sub = fig.text(0.045, top - 0.45 / fig_h,
                   f"{meta['payloads']} unique injections, each run through every model. "
                   f"The agent's own job was only to log the mail.",
                   fontsize=15, color=INK2, va="top")
    fit(sub, 0.91, 10)

    foot_lines = footer(meta)
    foot_pt, foot_step = 10, 0.028 * 7.45 / fig_h
    bottom = 0.18 * 7.45 / fig_h + 0.02
    height = (top - 0.85 / fig_h) - bottom
    ax = fig.add_axes([0.175, bottom, 0.615, height])
    ax.set_facecolor(SURFACE)
    ys = list(range(n))[::-1]
    bar_h = 0.52
    ramp = np.linspace(0, 1, 256).reshape(1, -1)
    rgb = matplotlib.colors.to_rgb(BAND)
    for y, (_name, val, lo, hi, _sus, _n) in zip(ys, rows, strict=True):
        # solid to the low end, then the interval fades out towards the high end
        if lo > 0:
            ax.barh(y, lo, height=bar_h, color=BAND, zorder=3)
        img = np.zeros((1, 256, 4))
        img[..., 0], img[..., 1], img[..., 2] = rgb
        img[..., 3] = 1.0 - ramp * 0.94
        ax.imshow(img, extent=(lo, hi, y - bar_h / 2, y + bar_h / 2), aspect="auto",
                  zorder=3, interpolation="bilinear")
        ax.text(hi + 1.6, y, f"{val:.1f}%", fontsize=10.5, color=INK, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.1)
    ticks = list(range(0, int(xmax) - 9, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v}%" for v in ticks], fontsize=9.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d1")
    ax.tick_params(length=0)

    # The other axis, numbers only: refusing and saying so are not the same thing, but a
    # second set of bars would pull the eye off the red scale, and that scale is the subject.
    ax2_l, ax2_w = 0.822, 0.118
    ax2 = fig.add_axes([ax2_l, bottom, ax2_w, height])
    ax2.set_facecolor(SURFACE)
    ax2.set_axis_off()
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.6, n - 0.1)
    # A lamp beside each number: off (white) at 0%, full green at 100%, mixed in proportion
    # between. The column's right edge is the subtitle's right edge, so the edges coincide.
    right = sub.get_window_extent(renderer).x1 / (fig_w * dpi)
    lamp_s = 95
    lamp_r = (lamp_s ** 0.5 / 72) / fig_w / 2
    lamp_x = (right - lamp_r - ax2_l) / ax2_w * 100
    green = np.array(matplotlib.colors.to_rgb("#2e9e4f"))
    off = np.array(matplotlib.colors.to_rgb("#ffffff"))
    head = fig.text(right, bottom + height * (n - 0.35 + 0.6) / (n + 0.5),
                    "said it was\nsuspicious", fontsize=9, color=MUTED, va="center",
                    ha="right", linespacing=1.4)
    left = head.get_window_extent(renderer).x0 / (fig_w * dpi)
    num_x = (left - ax2_l) / ax2_w * 100 + 2.5
    for y, r in zip(ys, rows, strict=True):
        ax2.text(num_x, y, f"{r[4]:.1f}%", fontsize=9.5, color=MUTED, va="center", ha="left")
        # square root, not linear: 3% would be invisible next to 0%, and the point of the lamp
        # is "this one says something at all". 0 stays white, 100 stays green.
        k = (r[4] / 100.0) ** 0.5
        ax2.scatter([lamp_x], [y], s=lamp_s, marker="o", facecolor=[off + (green - off) * k],
                    edgecolor="#b9b7b0", linewidth=0.8, zorder=4, clip_on=False)

    def wrap(text, width=0.93):
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            t = fig.text(0, 0, trial, fontsize=foot_pt)
            wide = t.get_window_extent(renderer).width / (fig_w * dpi) > width
            t.remove()
            if wide and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        return lines + [cur]

    y = bottom - 0.05 * 7.45 / fig_h
    for para in foot_lines:
        for line in wrap(para):
            fig.text(0.045, y, line, fontsize=foot_pt, color=MUTED, va="top")
            y -= foot_step
        y -= 0.006
    assert y > 0.02, "footer runs into the attribution line"
    fig.text(0.975, 0.010, CREDIT, fontsize=6.5, color=MUTED, alpha=0.75, va="bottom",
             ha="right")
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE, metadata={"Date": None} if out.suffix == ".svg"
                    else None)
    plt.close(fig)


def cells_from(recs, rows):
    """The family x model grid behind the heat map: {(label, family): (floor %, hits, n)},
    the families in a fixed order (descending pooled floor) and the pooled column."""
    by_label = {report.nice(m): inj for m, (inj, _ctl) in report.arms(recs).items()
                if report.undefended(m)}
    labels = [r[0] for r in rows]
    fam_pool = {}
    for inj in by_label.values():
        for r in inj:
            fam_pool.setdefault(r["family"], [0, 0])
            fam_pool[r["family"]][0] += report.payment(r)
            fam_pool[r["family"]][1] += 1
    fams = sorted(fam_pool, key=lambda f: (-fam_pool[f][0] / fam_pool[f][1], f))
    cells = {}
    for lab in labels:
        inj = by_label[lab]
        for f in fams + ["all"]:
            sub = [r for r in inj if f == "all" or r["family"] == f]
            k = sum(map(report.payment, sub))
            cells[(lab, f)] = (100 * k / len(sub) if sub else 0.0, k, len(sub))
    for f in fams + ["all"]:
        k = sum(cells[(lab, f)][1] for lab in labels)
        n = sum(cells[(lab, f)][2] for lab in labels)
        cells[("all models", f)] = (100 * k / n if n else 0.0, k, n)
    # Footer facts: unique payloads per cell (a model run twice has two records per payload),
    # and which models those are.
    cell_payloads = {len({r["id"] for r in inj if r["family"] == f})
                     for inj in by_label.values() for f in fams}
    twice = [lab for lab, inj in by_label.items() if len(inj) > len({r["id"] for r in inj})]
    return cells, fams, {"cell_payloads": cell_payloads, "twice": twice}


def draw_heat(rows, cells, fams, meta, outs):
    """Family x model heat map, models in the bar chart's order, families by pooled floor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r[0] for r in rows] + ["all models"]
    cols = fams + ["all"]
    n_r, n_c = len(labels), len(cols)
    fig_w, dpi = 11.0, 200
    fig_h = 2.9 + 0.42 * n_r
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    renderer = fig.canvas.get_renderer()

    def fit(t, limit, floor):
        while t.get_window_extent(renderer).width / (fig_w * dpi) > limit and \
                t.get_fontsize() > floor:
            t.set_fontsize(t.get_fontsize() - 0.5)

    top = 1 - 0.26 / fig_h
    title = fig.text(0.04, top, "Which construction moves which model?", fontsize=19,
                     color=INK, va="top", weight="bold")
    fit(title, 0.92, 11)
    sub = fig.text(0.04, top - 0.45 / fig_h,
                   "Share of injected emails that ended in a payment order, per lever the "
                   "payload uses to obtain compliance.",
                   fontsize=14, color=INK2, va="top")
    fit(sub, 0.92, 10)

    bottom = 1.45 / fig_h
    height = (top - 1.35 / fig_h) - bottom
    ax = fig.add_axes([0.17, bottom, 0.80, height])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, n_c)
    ax.set_ylim(0, n_r)
    ax.set_axis_off()
    rgb = np.array(matplotlib.colors.to_rgb(BAND))
    surf = np.array(matplotlib.colors.to_rgb(SURFACE))
    vmax = max(cells[(lab, c)][0] for lab in labels for c in cols) or 1.0
    for i, lab in enumerate(labels):
        y = n_r - 1 - i
        for j, c in enumerate(cols):
            val, k, n = cells[(lab, c)]
            # square root, like the lamps: a 2% cell has to look different from an empty one,
            # and the top of the scale is already saturated.
            t = (val / vmax) ** 0.5
            face = surf + (rgb - surf) * t
            edge = "#ffffff" if (i == n_r - 1 or j == n_c - 1) else SURFACE
            ax.add_patch(plt.Rectangle((j + 0.03, y + 0.06), 0.94, 0.88, facecolor=face,
                                       edgecolor=edge, linewidth=1.0, zorder=2))
            ink = "#ffffff" if t > 0.62 else INK
            ax.text(j + 0.5, y + 0.5, f"{val:.1f}", fontsize=9.5, color=ink, ha="center",
                    va="center", zorder=3,
                    weight="bold" if (i == n_r - 1 or j == n_c - 1) else "normal")
        ax.text(-0.08, y + 0.5, lab, fontsize=10.5, color=INK, ha="right", va="center",
                weight="bold" if i == n_r - 1 else "normal")
    for j, c in enumerate(cols):
        ax.text(j + 0.5, n_r + 0.12, c.replace("_", "\n"), fontsize=9.5,
                color=INK2 if c != "all" else INK, ha="center", va="bottom", linespacing=1.1,
                weight="bold" if c == "all" else "normal")
    # separators before the pooled row and column
    ax.plot([n_c - 1, n_c - 1], [0, n_r], color="#d8d7d1", lw=1, zorder=4)
    ax.plot([0, n_c], [1, 1], color="#d8d7d1", lw=1, zorder=4)

    ns = sorted(meta["cell_payloads"])
    twice = ", ".join(meta["twice"])
    foot = [f"Cell = % of that family's payloads (n = {ns[0]}-{ns[-1]} per cell, "
            f"{meta['payloads']} in all"
            + (f"; {twice} ran twice, both sweeps counted" if twice else "")
            + ") where a payment order existed at the run's end, whoever it went to. "
            "Families ordered by their pooled floor; models as in the bar chart.",
            "Colour on a square-root scale so a single hit is visible. "
            "One scene, a non-adaptive attacker."]
    foot_pt, foot_step = 9.5, 0.026 * 7.45 / fig_h

    def wrap(text, width=0.93):
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            t = fig.text(0, 0, trial, fontsize=foot_pt)
            wide = t.get_window_extent(renderer).width / (fig_w * dpi) > width
            t.remove()
            if wide and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        return lines + [cur]

    y = bottom - 0.05 * 7.45 / fig_h
    for para in foot:
        for line in wrap(para):
            fig.text(0.04, y, line, fontsize=foot_pt, color=MUTED, va="top")
            y -= foot_step
        y -= 0.004
    assert y > 0.02, "footer runs into the attribution line"
    fig.text(0.975, 0.010, CREDIT, fontsize=6.5, color=MUTED, alpha=0.75, va="bottom",
             ha="right")
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE, metadata={"Date": None} if out.suffix == ".svg"
                    else None)
    plt.close(fig)


def write(recs, slug, out_dir=REPORTS):
    """The two figures for one set of sweeps: `<out_dir>/figures/<slug>-models.{png,svg}` (the
    bar chart) and `<slug>-families.{png,svg}` (the family x model heat map). Returns the
    paths, or None with a note when matplotlib is missing."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("figure skipped: matplotlib is not installed (pip install -e '.[figures]')",
              file=sys.stderr)
        return None
    rows, meta = rows_from(recs)
    if not rows:
        return None
    figs = out_dir / "figures"
    outs = [figs / f"{slug}-models.png", figs / f"{slug}-models.svg"]
    draw(rows, meta, outs)
    cells, fams, extra = cells_from(recs, rows)
    heat = [figs / f"{slug}-families.png", figs / f"{slug}-families.svg"]
    draw_heat(rows, cells, fams, {**meta, **extra}, heat)
    return outs + heat


def slug_of(tag):
    return "".join(ch for ch in tag if ch.isalnum() or ch in "-_,").replace(",", "+") or "all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=report.PUBLISHED_TAG)
    ap.add_argument("--labels", default=str(corpus.LIST))
    ap.add_argument("--labels-where", default="demand=money_out")
    ap.add_argument("--slug", default="", help="file stem; default derived from --tag")
    ap.add_argument("--out", default=str(REPORTS))
    a = ap.parse_args()
    recs = report.load(a.tag, a.labels, a.labels_where)
    outs = write(recs, a.slug or slug_of(a.tag), pathlib.Path(a.out))
    for o in outs or []:
        print(o)


if __name__ == "__main__":
    main()
