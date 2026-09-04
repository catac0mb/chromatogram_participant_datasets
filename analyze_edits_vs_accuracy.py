#!/usr/bin/env python3
"""
Do people who make MORE edits get BETTER accuracy?  (continuous, not binary)

Instead of splitting into editors vs non-editors, this treats editing intensity as
a continuous predictor and asks whether it tracks accuracy. This answers "do heavier
editors do better?" with more power and no arbitrary >=1-edit cutoff.

EDIT INTENSITY (per participant, chosen with --edit-measure):
  rate      edits per visited chart              (default; normalises for how many
                                                   charts a person completed)
  total     total edit actions
  unique    distinct AI peaks edited per chart   (breadth, ignores repeat nudges)
So a "bare-minimum" rubber-stamper has a low value and a heavy reviser a high one.

ACCURACY (per participant): mean F1 / precision / recall / mean IoU over visited
charts with ground truth.

TESTS
  1. Correlation of edit intensity with each accuracy outcome, run POOLED and
     separately WITHIN each AI condition (to see if a relationship holds only in
     some conditions). PRIMARY test = Spearman rank correlation (non-parametric —
     no normality assumption, robust to skew/outliers, monotonic); Pearson reported
     alongside. Effect size = the coefficient ρ itself (|ρ|: <.1 negligible, <.3
     small, <.5 medium, ≥.5 large) with an approximate 95% CI. Correction =
     Benjamini-Hochberg FDR across the 4 outcomes, within each condition group.
  2. Regression = OLS with HC3 heteroskedasticity-robust standard errors (accuracy
     is bounded in [0,1], so residual variance is non-constant). Pooled model
     outcome ~ edit_intensity + confidence*threshold; per-condition model
     outcome ~ edit_intensity (2x2 factors are constant within a condition).
     Reported: raw slope, robust p, standardised slope β, R².
  3. Robustness (chart level, pooled): mixed model f1 ~ edit_count + trial_c, random
     participant intercept + chromatogram VC.
  Also prints accuracy by edit TERCILE (low/med/high) to reveal non-monotonic patterns.

FIGURES
  edits_accuracy_correlation_forest.png  Spearman ρ ± 95% CI per condition, per
                                         outcome (red ◆ = significant) — see at a
                                         glance whether the effect is only in some conditions
  edits_vs_accuracy_scatter.png          scatter with a regression line per condition
  edits_accuracy_terciles.png            mean F1 by low/med/high edit tercile

Needs exclusions.py, factorial_common.py, score_peaks.py. USAGE:


  python3 analyze_edits_vs_accuracy.py --submissions AI_comparisons_participants \
    --ground-truth DATA_DIR/synthetic_data --edit-measure rate \
        --plots AI_comparisons_edit_acc_figs --csv edits_acc.csv \
            --edit-events annotation 

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
from score_peaks import (load_ground_truth, chrom_stem, submitted_triples,  # noqa: E402
                         compute_metrics, was_visited)

OUTCOMES = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
            ("mean_iou", "Mean IoU (matched)")]
MEASURE_LABEL = {"rate": "edits per visited chart", "total": "total edits",
                 "unique": "unique peaks edited per chart"}


def spearman_ci(rho, n, conf=0.95):
    """Approximate 95% CI for Spearman rho via Fisher z (Bonett-Wright SE)."""
    if n is None or n <= 4 or rho is None or np.isnan(rho):
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(rho, -0.999, 0.999))
    se = 1.03 / np.sqrt(n - 3)                       # Bonett-Wright factor for Spearman
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    return (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))


def effect_label(r):
    """Cohen's rough benchmarks for a correlation coefficient."""
    a = abs(r)
    return ("negligible" if a < 0.10 else "small" if a < 0.30
            else "medium" if a < 0.50 else "large")


def load(subs, gtdir, measure, args):
    gt = load_ground_truth(gtdir)
    if not gt:
        print(f"No ground truth under {gtdir}"); sys.exit(1)
    records, chart_rows = [], []
    for path in sorted(glob.glob(os.path.join(subs, "*.json"))):
        try:
            doc = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip {os.path.basename(path)}: {e}", file=sys.stderr); continue
        data = doc.get("data", doc)
        if not isinstance(data, dict) or "chromatograms" not in data:
            continue
        chroms = data.get("chromatograms") or []
        pid = data.get("prolificPid") or data.get("userName") or path
        cond = data.get("visualizationMode")
        pos = fc.trial_position_map(data)
        per_ec, tot_edits, from_log = fc.edit_counts(data, args.edit_events)
        f1s, precs, recs, ious = [], [], [], []
        n_visited = 0; tot_unique = 0
        for i, ch in enumerate(chroms, start=1):
            if not was_visited(ch):
                continue
            n_visited += 1
            ec = per_ec.get(i - 1, 0)          # editLog chromIdx is 0-based; array index = i-1
            uq = ch.get("uniquePeaksEdited") or 0
            tot_unique += uq
            g = gt.get(chrom_stem(ch.get("file")))
            if g is None:
                continue
            m = compute_metrics(submitted_triples(ch.get("annotations")), g)
            f1s.append(m["f1"]); precs.append(m["precision"])
            recs.append(m["recall"]); ious.append(m["mean_iou"])
            chart_rows.append(dict(participant_id=pid, condition=cond,
                                   chromatogram_id=chrom_stem(ch.get("file")),
                                   trial=pos.get(chrom_stem(ch.get("file")), i),
                                   edit_count=ec, unique_edited=uq, f1=m["f1"]))
        if measure == "total":
            intensity = tot_edits
        elif measure == "unique":
            intensity = (tot_unique / n_visited) if n_visited else np.nan
        else:  # rate
            intensity = (tot_edits / n_visited) if n_visited else np.nan
        row = {"participant_id": pid, "condition": cond, "edit_intensity": intensity,
               "n_edits": tot_edits, "n_visited": n_visited,
               "f1": np.nanmean(f1s) if f1s else np.nan,
               "precision": np.nanmean(precs) if any(~np.isnan(precs)) else np.nan,
               "recall": np.nanmean(recs) if recs else np.nan,
               "mean_iou": np.nanmean(ious) if any(~np.isnan(ious)) else np.nan}
        ev = exclusions.evaluate(data, n_visited, args)
        records.append(dict(key=pid, fname=os.path.basename(path),
                            userName=data.get("userName"), condition=cond,
                            dur=data.get("sessionDurationMs"), row=row, **ev))
    if not args.keep_duplicates:
        kept, dropped = exclusions.resolve_duplicates(records)
        dpids = {r["key"] for r in dropped if r["excluded"]}
        chart_rows = [r for r in chart_rows if r["participant_id"] not in dpids]
        records = kept + dropped
    exclusions.print_audit(records, args)
    df = pd.DataFrame([r["row"] for r in records if not r["excluded"]])
    cdf = pd.DataFrame(chart_rows)
    return df, cdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--edit-measure", choices=["rate", "total", "unique"], default="rate")
    ap.add_argument("--edit-events", choices=["annotation", "interaction", "boundary", "deletions", "all"],
                    default="annotation",
                    help="which editLog events count as an 'edit': annotation = boundary "
                         "drags + added + deleted peaks (default, recommended); boundary = "
                         "drags only; deletions = deletes only; all = every logged interaction "
                         "(= raw editCount, also counts selects/badge-clicks/pans)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--csv")
    ap.add_argument("--plots", nargs="?", const="plots", default=None, metavar="DIR")
    exclusions.add_exclusion_args(ap)
    args = ap.parse_args()

    df, cdf = load(args.submissions, args.ground_truth, args.edit_measure, args)
    if df.empty:
        print("No participants."); sys.exit(1)
    df = fc.add_factors(df)
    mlabel = MEASURE_LABEL[args.edit_measure]
    conds = [c for c in fc.COND_ORDER if c in set(df["condition"])]

    print("\n" + "=" * 92)
    print(f"DO MORE EDITS -> BETTER ACCURACY?   (edit intensity = {mlabel})")
    print("=" * 92)
    _EV = {"annotation": "boundary drags + added peaks + deleted peaks",
           "interaction": "all peak interactions except panning (selects, badge-clicks, "
                          "drags, adds, deletes, restores)",
           "boundary": "boundary drags only", "deletions": "deletions only",
           "all": "ALL logged interactions incl. selects/badge-clicks/pans (raw editCount)"}
    print(f"'edit' = {_EV.get(args.edit_events, args.edit_events)}  (--edit-events {args.edit_events})")
    d = df.dropna(subset=["edit_intensity"])
    print(f"{len(d)} participants across {len(conds)} conditions. "
          f"edit intensity: mean={d['edit_intensity'].mean():.2f}, "
          f"median={d['edit_intensity'].median():.2f}, "
          f"range=[{d['edit_intensity'].min():.2f}, {d['edit_intensity'].max():.2f}]")

    def corr_rows(sub):
        """Spearman (primary) + Pearson per outcome, BH-FDR across the 4 outcomes."""
        rows, ps = [], []
        for key, label in OUTCOMES:
            s = sub[["edit_intensity", key]].dropna()
            if len(s) < 6 or s[key].nunique() < 2 or s["edit_intensity"].nunique() < 2:
                rows.append(dict(key=key, label=label, n=len(s), rho=np.nan, rp=np.nan,
                                 r=np.nan, pp=np.nan)); ps.append(np.nan); continue
            rho, pp = stats.spearmanr(s["edit_intensity"], s[key])
            r, rp = stats.pearsonr(s["edit_intensity"], s[key])
            lo, hi = spearman_ci(rho, len(s))
            rows.append(dict(key=key, label=label, n=len(s), rho=rho, pp=pp, lo=lo, hi=hi,
                             r=r, rp=rp))
            ps.append(pp)
        for row, pa in zip(rows, fc.bh_fdr(ps)):
            row["p_fdr"] = pa
        return rows

    # ---- (1) correlations: pooled AND within each condition ----
    print("\n(1) CORRELATION of edit intensity with accuracy")
    print("    Primary test: SPEARMAN rank correlation (non-parametric; no normality")
    print("    assumption; robust to skew/outliers; detects monotonic association).")
    print("    Effect size = rho itself (|rho|: <.1 negligible, <.3 small, <.5 medium, else large).")
    print("    Correction: Benjamini-Hochberg FDR across the 4 outcomes, within each row-group.")
    forest = {}   # (group,label) -> row, for the figure
    for gname, sub in [("ALL AI conditions (pooled)", d)] + [(c, d[d["condition"] == c]) for c in conds]:
        print(f"\n  [{gname}]  n={sub['edit_intensity'].notna().sum()}")
        print(f"    {'outcome':20}{'Spearman ρ':>12}{'95% CI':>18}{'p':>9}{'p_FDR':>9}"
              f"   {'effect':10} {'Pearson r':>10}")
        for row in corr_rows(sub):
            forest[(gname, row["label"])] = row
            if np.isnan(row["rho"]):
                print(f"    {row['label']:20}{'—':>12}{'':>18}{'':>9}{'':>9}   (too few)"); continue
            sig = fc.stars(row["p_fdr"])
            ci = f"[{row['lo']:+.2f},{row['hi']:+.2f}]"
            print(f"    {row['label']:20}{row['rho']:>+12.3f}{ci:>18}{row['pp']:>9.4f}"
                  f"{row['p_fdr']:>9.4f}{sig:>4} {effect_label(row['rho']):10} {row['r']:>+10.3f}")

    # ---- (2) regression, HC3 robust SE ----
    print("\n(2) REGRESSION — ordinary least squares with HC3 heteroskedasticity-robust")
    print("    standard errors (chosen because accuracy is bounded in [0,1], so residual")
    print("    variance is non-constant; HC3 keeps the slope estimate but corrects its SE/p).")
    print("    POOLED model: outcome ~ edit_intensity + confidence*threshold  (adjusts for the 2x2).")
    print("    PER-CONDITION: outcome ~ edit_intensity  (the 2x2 factors are constant within a")
    print("    condition, so they are dropped). Reported: slope, raw robust p, p_FDR (BH-FDR")
    print("    across the 4 outcomes within each group), standardised β, R².")
    print("    ** significance is judged by p_FDR (corrected), not raw p. **")
    for gname, sub, formula in ([("ALL AI conditions (pooled)", d,
                                  "Q('{k}') ~ edit_intensity + confidence * threshold")]
                                + [(c, d[d["condition"] == c], "Q('{k}') ~ edit_intensity")
                                   for c in conds]):
        print(f"\n  [{gname}]   {'outcome':22}{'slope':>9}{'raw p':>9}{'p_FDR':>9}{'β':>8}{'R²':>7}")
        rowbuf, praw = [], []
        for key, label in OUTCOMES:
            s = sub.dropna(subset=[key, "edit_intensity"])
            if len(s) < 10 or s[key].nunique() < 2 or s["edit_intensity"].nunique() < 2:
                rowbuf.append((label, None)); praw.append(np.nan); continue
            ols = smf.ols(formula.format(k=key), data=s).fit(cov_type="HC3")
            b = ols.params.get("edit_intensity", np.nan)
            p = ols.pvalues.get("edit_intensity", np.nan)
            beta = b * s["edit_intensity"].std() / s[key].std() if s[key].std() else np.nan
            rowbuf.append((label, dict(b=b, p=p, beta=beta, r2=ols.rsquared))); praw.append(p)
        for (label, r), pa in zip(rowbuf, fc.bh_fdr(praw)):
            if r is None:
                print(f"    {label:22}   insufficient data"); continue
            print(f"    {label:22}{r['b']:>+9.4f}{r['p']:>9.4f}{pa:>9.4f}{fc.stars(pa):>4}"
                  f"{r['beta']:>+8.2f}{r['r2']:>7.3f}")

    # ---- (2b) DO THE SLOPES DIFFER BY CONDITION? ----
    print("\n(2b) DOES THE EDIT→ACCURACY RELATIONSHIP DIFFER BY CONDITION?")
    print("    Interaction model: outcome ~ edit_intensity * confidence * threshold (HC3).")
    print("    A significant edit_intensity:<factor> term = the slope depends on that feature.")
    print("    p_FDR = BH-FDR across the 4 outcomes, per interaction term. Judge on p_FDR.")
    term_keys = ["edit_intensity:confidence", "edit_intensity:threshold",
                 "edit_intensity:confidence:threshold"]
    short = {"edit_intensity:confidence": "×confidence",
             "edit_intensity:threshold": "×threshold",
             "edit_intensity:confidence:threshold": "×conf:thr"}
    raw = {t: [] for t in term_keys}; labels_ok = []
    for key, label in OUTCOMES:
        s = d.dropna(subset=[key, "edit_intensity"])
        if len(s) < 20 or s[key].nunique() < 2 or s["edit_intensity"].nunique() < 2:
            labels_ok.append((label, None))
            for t in term_keys:
                raw[t].append(np.nan)
            continue
        m = smf.ols(f"Q('{key}') ~ edit_intensity * confidence * threshold",
                    data=s).fit(cov_type="HC3")
        labels_ok.append((label, m))
        for t in term_keys:
            raw[t].append(m.pvalues.get(t, np.nan))
    adj = {t: fc.bh_fdr(raw[t]) for t in term_keys}
    print(f"\n    {'outcome':20}" + "".join(f"{short[t]+' (raw/FDR)':>26}" for t in term_keys))
    any_sig = False
    for i, (label, m) in enumerate(labels_ok):
        if m is None:
            print(f"    {label:20}   insufficient data"); continue
        cells = []
        for t in term_keys:
            rp, ap = raw[t][i], adj[t][i]
            if not np.isnan(ap) and ap < args.alpha:
                any_sig = True
            cells.append(f"{rp:.3f}/{ap:.3f}{fc.stars(ap)}")
        print(f"    {label:20}" + "".join(f"{c:>26}" for c in cells))
    print(f"    -> {'some edit×condition interaction survives correction' if any_sig else 'NO edit×condition interaction survives BH-FDR — the edit→accuracy slope does not differ by condition'}")

    # ---- (2c) pairwise difference in Spearman rho (Fisher r-to-z), primary outcome ----
    from itertools import combinations
    prim = OUTCOMES[0][1]
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
        sig = fc.stars(pa)
        if not np.isnan(pa) and pa < args.alpha:
            any_diff = True
        print(f"    {a} (ρ={ra:+.2f}) vs {b} (ρ={rb:+.2f}): z={z:+.2f}, p={p:.4f}, p_FDR={pa:.4f} {sig}")
    print(f"    -> {'some conditions differ' if any_diff else 'no condition-pair differs significantly'}"
          f" in the edit→{prim} relationship.")
    print("    NOTE: testing whether SLOPES differ needs ~4x the data of testing each slope;")
    print("    a null here means 'no detectable difference', not 'the slopes are equal'.")

    # tercile means for F1
    print("\n(3) Mean F1 by edit-intensity tercile (reveals non-monotonic patterns):")
    s = d.dropna(subset=["f1", "edit_intensity"]).copy()
    if len(s) >= 12:
        try:
            s["tercile"] = pd.qcut(s["edit_intensity"], 3, labels=["low", "med", "high"])
            for t in ["low", "med", "high"]:
                v = s.loc[s["tercile"] == t, "f1"]
                print(f"   {t:6} edits: mean F1={v.mean():.3f}  (n={len(v)})")
        except ValueError:
            print("   (too many ties in edit intensity to form terciles)")

    # 4. chart-level robustness
    if not cdf.empty and cdf["participant_id"].nunique() >= 4:
        cdf = fc.add_factors(cdf); cdf["trial_c"] = cdf["trial"] - cdf["trial"].mean()
        print("\n(4) Chart-level mixed model: f1 ~ edit_count + trial_c "
              "(random participant + chromatogram):")
        res, dd, note = fc.fit_mixed(cdf.rename(columns={"edit_count": "edit_count"}),
                                     "f1", trial=True, chrom_type=False)
        # refit with edit_count as a predictor explicitly
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = smf.mixedlm("f1 ~ edit_count + trial_c", data=cdf,
                                groups=cdf["participant_id"],
                                vc_formula={"chromatogram": "0 + C(chromatogram_id)"})
                r2 = m.fit(reml=True, method="lbfgs")
                b = r2.params.get("edit_count", np.nan); p = r2.pvalues.get("edit_count", np.nan)
                print(f"   edit_count slope={b:+.5f}  p={p:.4f} {fc.stars(p)}  "
                      f"({'editing a chart → higher F1' if b > 0 else 'editing a chart → lower F1'})")
            except Exception as e:  # noqa: BLE001
                print(f"   (chart-level model did not fit: {e})")

        # 5. treatment-interaction chart-level mixed model — does the edit->F1 slope
        #    DIFFER by condition, with all the proper controls?
        print("\n(5) TREATMENT-INTERACTION mixed model (does the edit→F1 trend differ by")
        print("    condition?):  f1 ~ edit_count * confidence * threshold + trial_c,")
        print("    random participant intercept (some people better) + chromatogram variance")
        print("    component (some charts harder) + trial_c (order/fatigue).")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = smf.mixedlm("f1 ~ edit_count * confidence * threshold + trial_c",
                                data=cdf, groups=cdf["participant_id"],
                                vc_formula={"chromatogram": "0 + C(chromatogram_id)"})
                r3 = m.fit(reml=True, method="lbfgs")
                inter = [t for t in ["edit_count:confidence", "edit_count:threshold",
                                     "edit_count:confidence:threshold"] if t in r3.params.index]
                inter_fdr = dict(zip(inter, fc.bh_fdr([r3.pvalues[t] for t in inter])))
                print("    key terms (Wald tests; interaction p_FDR = BH across the 3 interaction terms):")
                for t in ["edit_count", "edit_count:confidence", "edit_count:threshold",
                          "edit_count:confidence:threshold", "trial_c"]:
                    if t in r3.params.index:
                        lab = ("edit slope (baseline=peaks_only)" if t == "edit_count"
                               else "order/fatigue" if t == "trial_c"
                               else "edit×" + t.split(":", 1)[1] + " (slope difference)")
                        if t in inter_fdr:
                            print(f"      {lab:42} coef={r3.params[t]:+.6f}  "
                                  f"raw p={r3.pvalues[t]:.4f}  p_FDR={inter_fdr[t]:.4f} "
                                  f"{fc.stars(inter_fdr[t])}")
                        else:
                            print(f"      {lab:42} coef={r3.params[t]:+.6f}  "
                                  f"p={r3.pvalues[t]:.4f} {fc.stars(r3.pvalues[t])}")
                any_int = any(not np.isnan(inter_fdr[t]) and inter_fdr[t] < args.alpha for t in inter)
                print(f"    -> {'the edit→F1 slope DIFFERS by condition (survives FDR)' if any_int else 'no evidence the edit→F1 slope differs by condition (no interaction survives FDR)'}")
                print("       (main edit slope + order/fatigue are single Wald tests, shown uncorrected;")
                print("        detecting slope DIFFERENCES needs much more data than the main slope).")
            except Exception as e:  # noqa: BLE001
                print(f"    (treatment-interaction model did not fit: {e})")

    if args.csv:
        df.to_csv(args.csv, index=False); print(f"\nTable -> {args.csv}")

    if args.plots is not None:
        os.makedirs(args.plots, exist_ok=True)
        groups = ["ALL AI conditions (pooled)"] + conds
        # --- Figure A: correlation forest (Spearman rho +/- 95% CI) per group ---
        fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(4.6 * len(OUTCOMES), 0.7 + 0.5 * len(groups)),
                                 squeeze=False, sharey=True)
        ypos = np.arange(len(groups))[::-1]
        for ax, (key, label) in zip(axes[0], OUTCOMES):
            for y, g in zip(ypos, groups):
                row = forest.get((g, label))
                if not row or np.isnan(row.get("rho", np.nan)):
                    continue
                sig = not np.isnan(row["p_fdr"]) and row["p_fdr"] < args.alpha
                col = "#b30000" if sig else "#64748b"
                ax.plot([row["lo"], row["hi"]], [y, y], color=col, lw=2, zorder=2)
                ax.scatter([row["rho"]], [y], s=55 if sig else 38, color=col, zorder=3,
                           marker="D" if sig else "o", edgecolors="white", linewidths=0.8)
            ax.axvline(0, color="#94a3b8", lw=1, ls="--")
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel("Spearman ρ (edits vs accuracy)")
            ax.set_xlim(-1, 1); ax.grid(axis="x", alpha=0.25, ls=":"); ax.set_axisbelow(True)
        axes[0][0].set_yticks(ypos)
        axes[0][0].set_yticklabels([g.replace(" (pooled)", "\n(pooled)") for g in groups], fontsize=8)
        fig.suptitle(f"Edits vs accuracy: Spearman ρ by condition  (red ◆ = significant, "
                     f"BH-FDR; line = 95% CI)  [{mlabel}]", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pf = os.path.join(args.plots, "edits_accuracy_correlation_forest.png")
        fig.savefig(pf, dpi=150); plt.close(fig)

        # --- Figure B: scatter with a regression line PER CONDITION (shared x-axis) ---
        fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(5 * len(OUTCOMES), 4.8),
                                 squeeze=False, sharex=True)
        for ax, (key, label) in zip(axes[0], OUTCOMES):
            for c in conds:
                sub = d[d["condition"] == c][["edit_intensity", key]].dropna()
                col = fc.COND_COLORS.get(c, "#999")
                if len(sub) < 3:
                    continue
                ax.scatter(sub["edit_intensity"], sub[key], s=20, alpha=0.45, color=col)
                row = forest.get((c, label))
                rtag = f" (ρ={row['rho']:+.2f})" if row and not np.isnan(row.get("rho", np.nan)) else ""
                if sub["edit_intensity"].nunique() > 1 and sub[key].nunique() > 1:
                    b1, b0 = np.polyfit(sub["edit_intensity"], sub[key], 1)
                    xs = np.linspace(sub["edit_intensity"].min(), sub["edit_intensity"].max(), 40)
                    ax.plot(xs, b0 + b1 * xs, color=col, lw=2, label=f"{c}{rtag}")
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel(mlabel); ax.set_ylabel(label)
            ax.grid(alpha=0.25, ls=":"); ax.set_axisbelow(True); ax.legend(fontsize=6)
        fig.suptitle(f"Edit intensity vs accuracy — regression line per condition  [{mlabel}]",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p1 = os.path.join(args.plots, "edits_vs_accuracy_scatter.png")
        fig.savefig(p1, dpi=150); plt.close(fig)

        # --- Figure C: F1 by edit tercile (pooled) ---
        p2 = None
        st = d.dropna(subset=["f1", "edit_intensity"]).copy()
        if len(st) >= 12:
            try:
                st["tercile"] = pd.qcut(st["edit_intensity"], 3, labels=["low", "med", "high"])
                fig, ax = plt.subplots(figsize=(6.5, 5))
                ms, es, ns = [], [], []
                for t in ["low", "med", "high"]:
                    v = st.loc[st["tercile"] == t, "f1"]
                    ms.append(v.mean()); es.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                    ns.append(len(v))
                ax.bar(range(3), ms, yerr=es, color=["#fdd0a2", "#91bfdb", "#1a4a89"], alpha=0.9, capsize=4)
                ax.set_xticks(range(3))
                ax.set_xticklabels([f"low\n(n={ns[0]})", f"med\n(n={ns[1]})", f"high\n(n={ns[2]})"])
                ax.set_ylabel("mean F1"); ax.set_xlabel(f"edit intensity tercile ({mlabel})")
                ax.set_title("Accuracy by edit-intensity tercile\n(error bars 95% CI)", fontweight="bold")
                ax.grid(axis="y", alpha=0.25, ls=":"); ax.set_axisbelow(True); fig.tight_layout()
                p2 = os.path.join(args.plots, "edits_accuracy_terciles.png")
                fig.savefig(p2, dpi=150); plt.close(fig)
            except ValueError:
                pass
        print("\nFigures written:")
        for p in [pf, p1, p2]:
            if p:
                print(f"  {p}")

        # --- Figure D: separate scatter per CONDITION (rows) x outcome (cols) ---
        # shared x everywhere, shared y within each outcome column, so conditions are
        # directly comparable on identical scales.
        fig, axes = plt.subplots(len(conds), len(OUTCOMES),
                                 figsize=(4.2 * len(OUTCOMES), 3.6 * len(conds)),
                                 squeeze=False, sharex=True, sharey="col")
        for ri, c in enumerate(conds):
            for ci, (key, label) in enumerate(OUTCOMES):
                ax = axes[ri][ci]
                sub = d[d["condition"] == c][["edit_intensity", key]].dropna()
                col = fc.COND_COLORS.get(c, "#999")
                if len(sub) >= 3:
                    ax.scatter(sub["edit_intensity"], sub[key], s=18, alpha=0.5, color=col)
                    if sub["edit_intensity"].nunique() > 1 and sub[key].nunique() > 1:
                        b1, b0 = np.polyfit(sub["edit_intensity"], sub[key], 1)
                        xs = np.linspace(sub["edit_intensity"].min(), sub["edit_intensity"].max(), 40)
                        ax.plot(xs, b0 + b1 * xs, color="#b30000", lw=2)
                        row = forest.get((c, label), {})
                        rt = f"ρ={row['rho']:+.2f}" if row and not np.isnan(row.get("rho", np.nan)) else ""
                        ax.set_title(f"{c} — {label}  {rt}", fontsize=8.5, fontweight="bold")
                else:
                    ax.set_title(f"{c} — {label} (n<3)", fontsize=8.5)
                if ci == 0:
                    ax.set_ylabel(c, fontsize=9, fontweight="bold")
                if ri == len(conds) - 1:
                    ax.set_xlabel(mlabel, fontsize=8)
                ax.grid(alpha=0.25, ls=":"); ax.set_axisbelow(True)
        fig.suptitle(f"Edit intensity vs accuracy — one panel per condition  [{mlabel}]",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        pbc = os.path.join(args.plots, "edits_vs_accuracy_scatter_by_condition.png")
        fig.savefig(pbc, dpi=150); plt.close(fig)
        print(f"  {pbc}")

    print("\n" + "=" * 92)
    print("METHODS SUMMARY (paper-ready; models, justifications, corrections)")
    print("=" * 92)
    print("Predictor: edit intensity. Outcomes: F1 (primary), precision, recall, mean IoU.")
    print("(1) PRIMARY — Spearman rank correlation, pooled and within each condition. Chosen")
    print("    over Pearson because accuracy is bounded/skewed (non-normal). Effect size = ρ")
    print("    with 95% CI (Fisher-z). CORRECTED: BH-FDR across the 4 outcomes within each group.")
    print("(2) Regression — OLS with HC3 robust SEs (bounded outcome -> heteroskedastic). Pooled")
    print("    adjusts for the 2x2; per-condition models the edit slope alone. Standardised β, R².")
    print("    CORRECTED: BH-FDR across the 4 outcomes within each group. (β≈Pearson r for a single")
    print("    predictor, so (2) is not an independent re-test of (1) — it is for covariate")
    print("    adjustment and standardised effect sizes; Spearman is primary.)")
    print("(2b) Treatment-interaction OLS (edit*confidence*threshold, HC3): does the slope differ")
    print("    by condition? CORRECTED: BH-FDR across the 4 outcomes, per interaction term.")
    print("(2c) Pairwise Fisher r-to-z of the Spearman ρ between conditions, BH-FDR across pairs.")
    print("(4) Chart-level mixed model f1 ~ edit_count + trial_c (random participant intercept +")
    print("    chromatogram variance component): within-participant, difficulty-controlled test")
    print("    that editing relates to accuracy. Wald p (single test).")
    print("(5) Treatment-interaction mixed model f1 ~ edit_count*confidence*threshold + trial_c")
    print("    (same random structure): does the within-participant slope differ by condition?")
    print("    Interaction terms Wald-tested, CORRECTED: BH-FDR across the 3 interaction terms.")
    print("")
    print("WHICH P-VALUES ARE CORRECTED: sections (1),(2),(2b),(2c),(5-interactions) report BH-FDR")
    print("and judge significance on it; raw p shown alongside. Single-coefficient Wald tests (the")
    print("main edit slope in (4)/(5), trial) are shown uncorrected as they are one test each.")
    print("")
    print("CAUSAL FOOTING: treatment is randomised (causal); edit intensity is NOT randomised, so")
    print("edit->accuracy links are associational. The mixed model removes participant-skill")
    print("confounding (within-person), but not chart-level reverse causation (people may edit more")
    print("on tractable charts). Report as a robust association, consistent across conditions.")


if __name__ == "__main__":
    main()