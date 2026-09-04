#!/usr/bin/env python3
"""
Do people who make MORE edits report more/less ENGAGEMENT and score differently on
NASA-TLX?  (continuous edit intensity, not a binary editor/non-editor split)

EDIT INTENSITY per participant (--edit-measure):
  rate    edits per visited chart  (default; normalises for charts completed)
  total   total edit actions
  unique  distinct AI peaks edited per chart

OUTCOMES (each once per participant):
  ENGAGEMENT items (raw units; direction differs and is labelled):
    fb_focus(+), fb_careful(+), fb_ready_stop(−), fb_repetitive(−), fb_ueq_boring_exciting(+)
  NASA-TLX: mental/physical/temporal demand, performance, effort, frustration, overall

TESTS
  1. Correlation of edit intensity with each outcome — Pearson (linear) and Spearman
     (monotonic), BH-FDR across items within each family (engagement / TLX separately).
  2. OLS  outcome ~ edit_intensity + confidence*threshold  (edit slope, adjusted for
     condition), for each outcome.
  Also prints outcome means by edit TERCILE (low/med/high editors).

FIGURES (median split: fewer vs more edits, grouped-bar histograms)
  edits_engagement_histogram.png   engagement-item distributions, low vs high editors
  edits_tlx_histogram.png          NASA-TLX subscale distributions, low vs high editors

Needs exclusions.py, factorial_common.py. 

USAGE:

  python3 analyze_edits_vs_survey.py AI_comparison_participants \
    --plots edits_survey_figs --csv edits_survey.csv --edit-events annotation

    options for --edit-events:
    --edit-events annotation = boundary drags + added peaks + deleted peaks (default, recommended)
    --edit-events boundary = boundary drags only
    --edit-events deletions = deletions only
    --edit-events all = every logged interaction (raw editCount, also counts selects/badge-clicks/pans)
    --edits-events interaction = every logged interaction (raw editCount, also counts selects/badge-clicks/pans)
"""

import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exclusions, factorial_common as fc  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

ENGAGE = [("fb_focus", "Q1 focus", (1, 5), "+"),
          ("fb_careful", "Q2 careful", (1, 5), "+"),
          ("fb_ready_stop", "Q3 ready-stop", (1, 5), "-"),
          ("fb_repetitive", "Q5 repetitive", (1, 5), "-"),
          ("fb_ueq_boring_exciting", "Q8 boring→exciting", (1, 7), "+")]
TLX = [("mentalDemand", "Mental Demand"), ("physicalDemand", "Physical Demand"),
       ("temporalDemand", "Temporal Demand"), ("performance", "Performance"),
       ("effort", "Effort"), ("frustration", "Frustration"),
       ("overallWorkload", "Overall Workload")]
SUBS = [k for k, _ in TLX if k != "overallWorkload"]


def was_visited(ch):
    return ((ch.get("visitCount") or 0) > 0 or (ch.get("totalActiveMs") or 0) > 0
            or ch.get("finishedAtMs") is not None)


def load(folder, measure, args):
    records = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            doc = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip {os.path.basename(path)}: {e}", file=sys.stderr); continue
        data = doc.get("data", doc)
        if not isinstance(data, dict):
            continue
        chroms = data.get("chromatograms") or []
        n_vis = sum(1 for c in chroms if was_visited(c))
        _perec, tot, _fromlog = fc.edit_counts(data, args.edit_events)
        uniq = sum((c.get("uniquePeaksEdited") or 0) for c in chroms if was_visited(c))
        if measure == "total":
            intensity = tot
        elif measure == "unique":
            intensity = (uniq / n_vis) if n_vis else np.nan
        else:
            intensity = (tot / n_vis) if n_vis else np.nan
        row = {"participant_id": data.get("prolificPid") or data.get("userName") or path,
               "condition": data.get("visualizationMode"), "edit_intensity": intensity}
        fb = ((data.get("surveys") or {}).get("feedback") or {}).get("responses") or {}
        for iid, *_ in ENGAGE:
            v = fb.get(iid)
            row[iid] = float(v) if isinstance(v, (int, float)) else np.nan
        tlx = (data.get("surveys") or {}).get("nasaTLX") or {}
        subs = tlx.get("subscaleScores") or {}
        for name, val in subs.items():
            if isinstance(val, dict) and not val.get("isAttentionCheck"):
                row[name] = val.get("score")
        ov = tlx.get("overallWorkload")
        if ov is None:
            vals = [row.get(s) for s in SUBS if row.get(s) is not None]
            ov = float(np.mean(vals)) if vals else None
        row["overallWorkload"] = ov
        ev = exclusions.evaluate(data, len(chroms), args)
        records.append(dict(key=row["participant_id"], fname=os.path.basename(path),
                            userName=data.get("userName"), condition=row["condition"],
                            dur=data.get("sessionDurationMs"), row=row, **ev))
    if not args.keep_duplicates:
        kept, dropped = exclusions.resolve_duplicates(records); records = kept + dropped
    exclusions.print_audit(records, args)
    return pd.DataFrame([r["row"] for r in records if not r["excluded"]])


def corr_block(df, items, title, alpha):
    print("\n" + "-" * 92); print(title); print("-" * 92)
    d = df.dropna(subset=["edit_intensity"])
    ps, rows = [], []
    for tup in items:
        key, label = tup[0], tup[1]
        direction = tup[3] if len(tup) > 3 else ""
        s = d[["edit_intensity", key]].dropna()
        if len(s) < 5 or s[key].nunique() < 2:
            print(f"  {label:22} insufficient data"); continue
        r, pr = stats.pearsonr(s["edit_intensity"], s[key])
        rho, psp = stats.spearmanr(s["edit_intensity"], s[key])
        ps.append(pr); rows.append((label, r, pr, rho, psp, len(s), direction))
    for (label, r, pr, rho, psp, n, dirn), pa in zip(rows, fc.bh_fdr(ps)):
        tag = {"+": "(↑=more engaged)", "-": "(↑=LESS engaged)"}.get(dirn, "")
        print(f"  {label:22} Pearson r={r:+.3f} (p={pr:.4f}, p_FDR={pa:.4f}){fc.stars(pa)}"
              f"   Spearman ρ={rho:+.3f}   n={n}  {tag}")
    return d


def reg_block(df, items, alpha):
    print("\n  OLS  outcome ~ edit_intensity + confidence*threshold  (edit slope):")
    for tup in items:
        key, label = tup[0], tup[1]
        s = df.dropna(subset=[key, "edit_intensity"])
        if len(s) < 12 or s[key].nunique() < 2:
            continue
        ols = smf.ols(f"Q('{key}') ~ edit_intensity + confidence * threshold", data=s).fit()
        b = ols.params.get("edit_intensity", np.nan); p = ols.pvalues.get("edit_intensity", np.nan)
        print(f"    {label:22} edit slope={b:+.4f}  p={p:.4f} {fc.stars(p)}")


def terciles(df, key, label):
    s = df.dropna(subset=[key, "edit_intensity"]).copy()
    if len(s) < 12:
        return
    try:
        s["t"] = pd.qcut(s["edit_intensity"], 3, labels=["low", "med", "high"])
    except ValueError:
        print(f"    {label}: too many ties for terciles"); return
    parts = [f"{t}={s.loc[s['t']==t, key].mean():.2f}(n={ (s['t']==t).sum() })"
             for t in ["low", "med", "high"]]
    print(f"    {label:22} " + "  ".join(parts))


def hist_by_editgroup(df, items, ranges, title, path, integer_bins):
    d = df.dropna(subset=["edit_intensity"]).copy()
    med = d["edit_intensity"].median()
    d["grp"] = np.where(d["edit_intensity"] > med, "more edits", "fewer edits")
    ncols = min(4, len(items)); nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), squeeze=False)
    axes = axes.ravel()
    groups = [("fewer edits", "#f19066"), ("more edits", "#2e6da4")]
    for ax, (key, label, rng) in zip(axes, [(k, l, r) for (k, l), r in zip(
            [(i[0], i[1]) for i in items], ranges)]):
        if integer_bins:
            lo, hi = rng; vals = list(range(lo, hi + 1)); edges = None
            bw = 0.8 / len(groups)
            for gi, (g, col) in enumerate(groups):
                v = d.loc[d["grp"] == g, key].dropna().values
                if not len(v):
                    continue
                props = [np.mean(np.round(v) == val) for val in vals]
                xs = [val + (gi - (len(groups) - 1) / 2) * bw for val in vals]
                ax.bar(xs, props, width=bw, color=col, alpha=0.9, edgecolor="white",
                       linewidth=0.5, label=f"{g} (n={len(v)})")
            ax.set_xticks(vals); ax.set_xlabel(f"response ({rng[0]}–{rng[1]})")
        else:
            edges = np.linspace(0, 100, 11); ct = (edges[:-1] + edges[1:]) / 2
            bw = (edges[1] - edges[0]) * 0.8 / len(groups)
            for gi, (g, col) in enumerate(groups):
                v = d.loc[d["grp"] == g, key].dropna().values
                if not len(v):
                    continue
                counts, _ = np.histogram(v, bins=edges)
                xs = ct + (gi - (len(groups) - 1) / 2) * bw
                ax.bar(xs, counts / len(v), width=bw, color=col, alpha=0.9,
                       edgecolor="white", linewidth=0.5, label=f"{g} (n={len(v)})")
            ax.set_xlabel("score (0–100)"); ax.set_xticks(edges[::2])
        ax.set_title(label, fontsize=10, fontweight="bold"); ax.set_ylabel("proportion")
        ax.grid(axis="y", alpha=0.25, ls=":"); ax.set_axisbelow(True); ax.legend(fontsize=7)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(title + f"  (median split at edit intensity = {med:.2f})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--edit-measure", choices=["rate", "total", "unique"], default="rate")
    ap.add_argument("--edit-events", choices=["annotation","interaction","boundary","deletions","all"],
                    default="annotation",
                    help="which editLog events count as an 'edit': annotation = boundary "
                         "drags + added + deleted peaks (default); boundary = drags only; "
                         "deletions = deletes only; all = every logged interaction")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--csv")
    ap.add_argument("--plots", nargs="?", const="plots", default=None, metavar="DIR")
    exclusions.add_exclusion_args(ap)
    args = ap.parse_args()

    df = fc.add_factors(load(args.folder, args.edit_measure, args))
    if df.empty:
        print("No participants."); sys.exit(1)

    print("\n" + "=" * 92)
    print("DO HEAVIER EDITORS REPORT DIFFERENT ENGAGEMENT / NASA-TLX?")
    print("=" * 92)
    d = df.dropna(subset=["edit_intensity"])
    print(f"{len(d)} participants. edit intensity ({args.edit_measure}): "
          f"mean={d['edit_intensity'].mean():.2f}, median={d['edit_intensity'].median():.2f}, "
          f"range=[{d['edit_intensity'].min():.2f}, {d['edit_intensity'].max():.2f}]")

    corr_block(df, ENGAGE, "(1a) ENGAGEMENT vs edit intensity (correlation, BH-FDR)", args.alpha)
    reg_block(df, [(i[0], i[1]) for i in ENGAGE], args.alpha)
    corr_block(df, TLX, "(1b) NASA-TLX vs edit intensity (correlation, BH-FDR)", args.alpha)
    reg_block(df, TLX, args.alpha)

    print("\n(2) Means by edit tercile:")
    print("   ENGAGEMENT:")
    for iid, label, *_ in ENGAGE:
        terciles(df, iid, label)
    print("   NASA-TLX:")
    for key, label in TLX:
        terciles(df, key, label)

    if args.csv:
        df.to_csv(args.csv, index=False); print(f"\nTable -> {args.csv}")

    if args.plots is not None:
        os.makedirs(args.plots, exist_ok=True)
        p1 = hist_by_editgroup(df, [(i[0], i[1]) for i in ENGAGE],
                               [i[2] for i in ENGAGE],
                               "Engagement items by editor group", 
                               os.path.join(args.plots, "edits_engagement_histogram.png"),
                               integer_bins=True)
        p2 = hist_by_editgroup(df, TLX, [(0, 100)] * len(TLX),
                               "NASA-TLX by editor group",
                               os.path.join(args.plots, "edits_tlx_histogram.png"),
                               integer_bins=False)
        print(f"\nFigures written:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
