#!/usr/bin/env python3
"""
Regression of NASA-TLX on how much people EDIT (continuous edit intensity), 2x2-aware.

NASA-TLX is collected once per participant, so this is a participant-level analysis
(no mixed model). It asks: do people who edit more report different workload?

EDIT INTENSITY per participant (--edit-measure): rate (edits/visited chart, default),
total (raw edits), unique (distinct peaks edited per chart).

OUTCOMES: the 6 NASA-TLX subscales + overall workload (0-100). Higher = more
workload (worse); 'performance' is already anchored low=better.

TESTS
  1. Association — SPEARMAN rank correlation (primary; non-parametric, no normality
     assumption, robust to skew), Pearson reported alongside. Run POOLED and WITHIN
     each AI condition. Effect size = rho (|rho|: <.1 negligible, <.3 small, <.5
     medium, >=.5 large) with an approximate 95% CI. Correction = Benjamini-Hochberg
     FDR across the 7 outcomes, within each condition group.
  2. REGRESSION — ordinary least squares with HC3 heteroskedasticity-robust standard
     errors (TLX is bounded 0-100 and skewed, so residual variance is non-constant;
     HC3 keeps the slope but corrects its SE/p). POOLED model adds the 2x2 covariates
     (workload ~ edit_intensity + confidence*threshold) so the edit slope is adjusted
     for condition; PER-CONDITION models are workload ~ edit_intensity alone (the 2x2
     factors are constant within a condition). Reported: raw slope (TLX points per
     unit edit intensity), robust p, standardised slope beta, R^2.

FIGURES
  edits_tlx_correlation_forest.png   Spearman rho +/- 95% CI per condition, per
                                     subscale (red = significant) — see where it holds
  edits_tlx_scatter.png              scatter with a regression line per condition

Needs exclusions.py, factorial_common.py. 

USAGE:

    python3 analyze_edits_vs_tlx.py AI_comparison_participants \
    --plots edits_tlx_figs --csv edits_tlx.csv --edit-events annotation
  
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
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exclusions, factorial_common as fc  # noqa: E402

SUBS = ["mentalDemand", "physicalDemand", "temporalDemand", "performance",
        "effort", "frustration"]
OUTCOMES = [("mentalDemand", "Mental Demand"), ("physicalDemand", "Physical Demand"),
            ("temporalDemand", "Temporal Demand"), ("performance", "Performance (low=better)"),
            ("effort", "Effort"), ("frustration", "Frustration"),
            ("overallWorkload", "Overall Workload")]
MEASURE_LABEL = {"rate": "edits per visited chart", "total": "total edits",
                 "unique": "unique peaks edited per chart"}


def was_visited(ch):
    return ((ch.get("visitCount") or 0) > 0 or (ch.get("totalActiveMs") or 0) > 0
            or ch.get("finishedAtMs") is not None)


def spearman_ci(rho, n, conf=0.95):
    if n is None or n <= 4 or rho is None or np.isnan(rho):
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(rho, -0.999, 0.999)); se = 1.03 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    return (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))


def effect_label(r):
    a = abs(r)
    return ("negligible" if a < .1 else "small" if a < .3 else "medium" if a < .5 else "large")


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
                            dur=data.get("sessionDurationMs"), row=row, has_tlx=bool(subs), **ev))
    if not args.keep_duplicates:
        kept, dropped = exclusions.resolve_duplicates(records); records = kept + dropped
    exclusions.print_audit(records, args)
    return pd.DataFrame([r["row"] for r in records if not r["excluded"] and r["has_tlx"]])


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
        print("No participants with NASA-TLX data."); sys.exit(1)
    mlabel = MEASURE_LABEL[args.edit_measure]
    conds = [c for c in fc.COND_ORDER if c in set(df["condition"])]
    d = df.dropna(subset=["edit_intensity"])

    print("\n" + "=" * 96)
    print(f"NASA-TLX vs EDIT INTENSITY   (edit intensity = {mlabel})")
    print("=" * 96)
    print(f"{len(d)} participants across {len(conds)} conditions. edit intensity: "
          f"mean={d['edit_intensity'].mean():.2f}, median={d['edit_intensity'].median():.2f}, "
          f"range=[{d['edit_intensity'].min():.2f}, {d['edit_intensity'].max():.2f}]")
    print("Higher TLX = more workload (worse). Positive slope = more editing → more workload.")

    def corr_rows(sub):
        rows, ps = [], []
        for key, label in OUTCOMES:
            s = sub[["edit_intensity", key]].dropna()
            if len(s) < 6 or s[key].nunique() < 2 or s["edit_intensity"].nunique() < 2:
                rows.append(dict(label=label, rho=np.nan)); ps.append(np.nan); continue
            rho, pp = stats.spearmanr(s["edit_intensity"], s[key])
            r, rp = stats.pearsonr(s["edit_intensity"], s[key])
            lo, hi = spearman_ci(rho, len(s))
            rows.append(dict(label=label, n=len(s), rho=rho, pp=pp, lo=lo, hi=hi, r=r))
            ps.append(pp)
        for row, pa in zip(rows, fc.bh_fdr(ps)):
            row["p_fdr"] = pa
        return rows

    # (1) correlations
    print("\n(1) CORRELATION (Spearman primary; BH-FDR across the 7 outcomes per group)")
    forest = {}
    for gname, sub in [("ALL AI conditions (pooled)", d)] + [(c, d[d["condition"] == c]) for c in conds]:
        print(f"\n  [{gname}]  n={sub['edit_intensity'].notna().sum()}")
        print(f"    {'outcome':24}{'Spearman ρ':>12}{'95% CI':>18}{'p_FDR':>9}   {'effect':10}{'Pearson r':>10}")
        for row in corr_rows(sub):
            forest[(gname, row["label"])] = row
            if np.isnan(row.get("rho", np.nan)):
                print(f"    {row['label']:24}{'—':>12}   (too few)"); continue
            sig = fc.stars(row["p_fdr"]); ci = f"[{row['lo']:+.2f},{row['hi']:+.2f}]"
            print(f"    {row['label']:24}{row['rho']:>+12.3f}{ci:>18}{row['p_fdr']:>9.4f}{sig:>4} "
                  f"{effect_label(row['rho']):10}{row['r']:>+10.3f}")

    # (2) regression HC3
    print("\n(2) REGRESSION — OLS with HC3 robust SEs")
    print("    POOLED: workload ~ edit_intensity + confidence*threshold  (adjusts for 2x2)")
    print("    PER-CONDITION: workload ~ edit_intensity")
    print("    Reported: slope (TLX pts per unit edit intensity), raw robust p, p_FDR")
    print("    (BH-FDR across the 7 subscales within each group), standardised β, R².")
    print("    ** significance is judged by p_FDR (corrected), not raw p. **")
    for gname, sub, formula in ([("ALL AI conditions (pooled)", d,
                                  "Q('{k}') ~ edit_intensity + confidence * threshold")]
                                + [(c, d[d["condition"] == c], "Q('{k}') ~ edit_intensity") for c in conds]):
        print(f"\n  [{gname}]   {'outcome':22}{'slope':>9}{'raw p':>9}{'p_FDR':>9}{'β':>8}{'R²':>7}")
        rowbuf, praw = [], []
        for key, label in OUTCOMES:
            s = sub.dropna(subset=[key, "edit_intensity"])
            if len(s) < 10 or s[key].nunique() < 2 or s["edit_intensity"].nunique() < 2:
                rowbuf.append((label, None)); praw.append(np.nan); continue
            ols = smf.ols(formula.format(k=key), data=s).fit(cov_type="HC3")
            b = ols.params.get("edit_intensity", np.nan); p = ols.pvalues.get("edit_intensity", np.nan)
            beta = b * s["edit_intensity"].std() / s[key].std() if s[key].std() else np.nan
            rowbuf.append((label, dict(b=b, p=p, beta=beta, r2=ols.rsquared))); praw.append(p)
        for (label, r), pa in zip(rowbuf, fc.bh_fdr(praw)):
            if r is None:
                print(f"    {label:22}   insufficient data"); continue
            print(f"    {label:22}{r['b']:>+9.4f}{r['p']:>9.4f}{pa:>9.4f}{fc.stars(pa):>4}"
                  f"{r['beta']:>+8.2f}{r['r2']:>7.3f}")

    # ---- (2b) does the edit->TLX relationship differ by condition? ----
    print("\n(2b) DOES THE EDIT→WORKLOAD RELATIONSHIP DIFFER BY CONDITION?")
    print("    NASA-TLX is measured ONCE per participant, so there is no chart/trial nesting")
    print("    and a mixed model does not apply here (nothing to nest). The correct test of")
    print("    'do the trends differ by condition' is the treatment-interaction regression:")
    print("    outcome ~ edit_intensity * confidence * threshold (HC3 robust SEs).")
    print("    A significant edit_intensity:<factor> term = the slope depends on that feature.")
    print("    p_FDR = BH-FDR across the 7 subscales, computed separately for each interaction")
    print("    term. ** Judge significance by p_FDR (corrected), not the raw p. **")
    # collect raw p per (outcome, term)
    term_keys = ["edit_intensity:confidence", "edit_intensity:threshold",
                 "edit_intensity:confidence:threshold"]
    raw = {t: [] for t in term_keys}; labels_ok = []
    for key, label in OUTCOMES:
        s = d.dropna(subset=[key, "edit_intensity"])
        if len(s) < 20 or s[key].nunique() < 2 or s["edit_intensity"].nunique() < 2:
            labels_ok.append((label, None)); 
            for t in term_keys:
                raw[t].append(np.nan)
            continue
        m = smf.ols(f"Q('{key}') ~ edit_intensity * confidence * threshold",
                    data=s).fit(cov_type="HC3")
        labels_ok.append((label, m))
        for t in term_keys:
            raw[t].append(m.pvalues.get(t, np.nan))
    adj = {t: fc.bh_fdr(raw[t]) for t in term_keys}
    short = {"edit_intensity:confidence": "×confidence",
             "edit_intensity:threshold": "×threshold",
             "edit_intensity:confidence:threshold": "×conf:thr"}
    print(f"\n    {'outcome':24}" + "".join(f"{short[t]+' (raw/FDR)':>26}" for t in term_keys))
    any_sig = False
    for i, (label, m) in enumerate(labels_ok):
        if m is None:
            print(f"    {label:24}   insufficient data"); continue
        cells = []
        for t in term_keys:
            rp, ap = raw[t][i], adj[t][i]
            if not np.isnan(ap) and ap < args.alpha:
                any_sig = True
            cells.append(f"{rp:.3f}/{ap:.3f}{fc.stars(ap)}")
        print(f"    {label:24}" + "".join(f"{c:>26}" for c in cells))
    print(f"    -> {'some edit×condition interaction survives correction' if any_sig else 'NO edit×condition interaction survives BH-FDR — the edit→workload slope does not differ by condition'}")

    # ---- (2c) pairwise difference in Spearman rho (Fisher r-to-z), primary = overall workload ----
    from itertools import combinations
    prim = "Overall Workload"
    print(f"\n(2c) PAIRWISE difference in Spearman ρ between conditions (Fisher r-to-z),")
    print(f"    primary outcome = {prim}; BH-FDR across the {len(list(combinations(conds,2)))} pairs:")
    pr, rows = [], []
    for a, b in combinations(conds, 2):
        ra, rb = forest.get((a, prim), {}), forest.get((b, prim), {})
        z, p = fc.fisher_z_diff(ra.get("rho"), ra.get("n"), rb.get("rho"), rb.get("n"))
        rows.append((a, b, ra.get("rho"), rb.get("rho"), z, p)); pr.append(p)
    any_diff = False
    for (a, b, ra, rb, z, p), pa in zip(rows, fc.bh_fdr(pr)):
        if p is None or np.isnan(p):
            continue
        if not np.isnan(pa) and pa < args.alpha:
            any_diff = True
        print(f"    {a} (ρ={ra:+.2f}) vs {b} (ρ={rb:+.2f}): z={z:+.2f}, p={p:.4f}, "
              f"p_FDR={pa:.4f} {fc.stars(pa)}")
    print(f"    -> {'some conditions differ' if any_diff else 'no condition-pair differs significantly'}"
          f" in the edit→workload relationship.")
    print("    NOTE: detecting a DIFFERENCE in slopes needs ~4x the data of detecting each slope;")
    print("    a null here means 'no detectable difference', not 'the slopes are equal'.")

    if args.csv:
        df.to_csv(args.csv, index=False); print(f"\nTable -> {args.csv}")

    if args.plots is not None:
        os.makedirs(args.plots, exist_ok=True)
        groups = ["ALL AI conditions (pooled)"] + conds
        # correlation forest
        fig, axes = plt.subplots(2, 4, figsize=(19, 6), squeeze=False)
        axes = axes.ravel(); ypos = np.arange(len(groups))[::-1]
        for ax, (key, label) in zip(axes, OUTCOMES):
            for y, g in zip(ypos, groups):
                row = forest.get((g, label))
                if not row or np.isnan(row.get("rho", np.nan)):
                    continue
                sig = not np.isnan(row["p_fdr"]) and row["p_fdr"] < args.alpha
                col = "#b30000" if sig else "#64748b"
                ax.plot([row["lo"], row["hi"]], [y, y], color=col, lw=2, zorder=2)
                ax.scatter([row["rho"]], [y], s=52 if sig else 36, color=col, zorder=3,
                           marker="D" if sig else "o", edgecolors="white", linewidths=0.8)
            ax.axvline(0, color="#94a3b8", lw=1, ls="--"); ax.set_xlim(-1, 1)
            ax.set_title(label, fontsize=9.5, fontweight="bold")
            ax.set_xlabel("Spearman ρ"); ax.set_yticks(ypos)
            ax.set_yticklabels([g.replace(" (pooled)", "") for g in groups], fontsize=7)
            ax.grid(axis="x", alpha=0.25, ls=":"); ax.set_axisbelow(True)
        for ax in axes[len(OUTCOMES):]:
            ax.axis("off")
        fig.suptitle(f"NASA-TLX vs edit intensity: Spearman ρ by condition  "
                     f"(red ◆ = significant, BH-FDR; line = 95% CI)  [{mlabel}]",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pf = os.path.join(args.plots, "edits_tlx_correlation_forest.png")
        fig.savefig(pf, dpi=150); plt.close(fig)
        # scatter with per-condition regression line (shared axes: all TLX are 0-100)
        fig, axes = plt.subplots(2, 4, figsize=(19, 8), squeeze=False,
                                 sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, (key, label) in zip(axes, OUTCOMES):
            for c in conds:
                sub = d[d["condition"] == c][["edit_intensity", key]].dropna()
                col = fc.COND_COLORS.get(c, "#999")
                if len(sub) < 3:
                    continue
                ax.scatter(sub["edit_intensity"], sub[key], s=16, alpha=0.4, color=col)
                row = forest.get((c, label))
                rtag = f" (ρ={row['rho']:+.2f})" if row and not np.isnan(row.get("rho", np.nan)) else ""
                if sub["edit_intensity"].nunique() > 1 and sub[key].nunique() > 1:
                    b1, b0 = np.polyfit(sub["edit_intensity"], sub[key], 1)
                    xs = np.linspace(sub["edit_intensity"].min(), sub["edit_intensity"].max(), 40)
                    ax.plot(xs, b0 + b1 * xs, color=col, lw=2, label=f"{c}{rtag}")
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel(mlabel); ax.set_ylabel(label); ax.set_ylim(0, 100)
            ax.grid(alpha=0.25, ls=":"); ax.set_axisbelow(True); ax.legend(fontsize=6)
        for ax in axes[len(OUTCOMES):]:
            ax.axis("off")
        fig.suptitle(f"NASA-TLX vs edit intensity — regression line per condition  [{mlabel}]",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        ps_ = os.path.join(args.plots, "edits_tlx_scatter.png")
        fig.savefig(ps_, dpi=150); plt.close(fig)
        # separate scatter per CONDITION (rows) x subscale (cols); shared axes so
        # conditions are directly comparable (all TLX are 0-100).
        fig, axes = plt.subplots(len(conds), len(OUTCOMES),
                                 figsize=(3.5 * len(OUTCOMES), 3.3 * len(conds)),
                                 squeeze=False, sharex=True, sharey=True)
        for ri, c in enumerate(conds):
            for ci, (key, label) in enumerate(OUTCOMES):
                ax = axes[ri][ci]
                sub = d[d["condition"] == c][["edit_intensity", key]].dropna()
                col = fc.COND_COLORS.get(c, "#999")
                if len(sub) >= 3:
                    ax.scatter(sub["edit_intensity"], sub[key], s=14, alpha=0.45, color=col)
                    if sub["edit_intensity"].nunique() > 1 and sub[key].nunique() > 1:
                        b1, b0 = np.polyfit(sub["edit_intensity"], sub[key], 1)
                        xs = np.linspace(sub["edit_intensity"].min(), sub["edit_intensity"].max(), 40)
                        ax.plot(xs, b0 + b1 * xs, color="#b30000", lw=2)
                        row = forest.get((c, label), {})
                        rt = f"ρ={row['rho']:+.2f}" if row and not np.isnan(row.get("rho", np.nan)) else ""
                        ax.set_title(f"{c} — {label} {rt}", fontsize=7.5, fontweight="bold")
                ax.set_ylim(0, 100)
                if ci == 0:
                    ax.set_ylabel(c, fontsize=8, fontweight="bold")
                if ri == len(conds) - 1:
                    ax.set_xlabel(mlabel, fontsize=7)
                ax.grid(alpha=0.25, ls=":"); ax.set_axisbelow(True)
        fig.suptitle(f"NASA-TLX vs edit intensity — one panel per condition  [{mlabel}]",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        pbc = os.path.join(args.plots, "edits_tlx_scatter_by_condition.png")
        fig.savefig(pbc, dpi=150); plt.close(fig)
        print(f"\nFigures written:\n  {pf}\n  {ps_}\n  {pbc}")

    print("\n" + "=" * 96)
    print("METHODS SUMMARY (paper-ready; every model, its justification, its correction)")
    print("=" * 96)
    print("Outcome: NASA-TLX (6 subscales + overall workload), each collected ONCE per")
    print("participant. Predictor: edit intensity (edits per visited chart). Unit of analysis")
    print("= participant, so all models are participant-level (a mixed model is NOT used, and")
    print("would be mis-specified, because there are no repeated TLX measurements to nest).")
    print("")
    print("(1) PRIMARY — Spearman rank correlation of edit intensity with each subscale, run")
    print("    pooled and within each condition. Chosen over Pearson because TLX is bounded")
    print("    (0-100) and skewed (non-normal); Spearman assumes no normality. Effect size = ρ")
    print("    with 95% CI (Fisher-z). CORRECTED: Benjamini-Hochberg FDR across the 7 subscales")
    print("    within each condition group.")
    print("(2) Regression — OLS with HC3 heteroskedasticity-robust SEs (bounded outcome ->")
    print("    non-constant residual variance; HC3 keeps the slope, corrects the SE/p). Pooled")
    print("    model adjusts for the 2x2 (confidence*threshold); per-condition models estimate")
    print("    the edit slope alone. Reported with standardised β and R². CORRECTED: BH-FDR")
    print("    across the 7 subscales within each group. Note: for a single predictor, β≈Pearson")
    print("    r, so (2) is not an independent confirmation of (1) — it is used for COVARIATE")
    print("    ADJUSTMENT (pooled) and standardised effect sizes, with Spearman as primary.")
    print("(2b) Treatment-interaction regression (edit_intensity * confidence * threshold, HC3)")
    print("    to test whether the edit->workload slope DIFFERS by condition. CORRECTED: BH-FDR")
    print("    across the 7 subscales, separately per interaction term. Significance judged on")
    print("    p_FDR. (Power to detect slope DIFFERENCES is ~4x lower than for main slopes, so a")
    print("    null = 'no detectable difference', not 'slopes equal'.)")
    print("(2c) Pairwise Fisher r-to-z comparisons of the Spearman ρ between conditions, BH-FDR")
    print("    across the 6 pairs (primary outcome = overall workload).")
    print("")
    print("WHICH P-VALUES ARE CORRECTED: sections (1), (2), (2b), (2c) all report BH-FDR-adjusted")
    print("p-values and judge significance on them. Raw p is shown alongside for transparency.")
    print("")
    print("CAUSAL FOOTING: treatment (confidence/threshold) is randomised, so treatment effects")
    print("are causal; edit intensity is NOT randomised, so edit->workload links are associational")
    print("and additionally confounded (editing IS effort, so a positive relationship may reflect")
    print("that doing more work feels like more work). Report as association, not cause.")


if __name__ == "__main__":
    main()
