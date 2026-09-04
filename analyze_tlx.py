#!/usr/bin/env python3
"""
NASA-TLX analysis across the four interface conditions.

WHAT IT DOES
------------
1. Extracts, per participant: the 6 NASA-TLX subscales + overall workload,
   the condition (visualizationMode), and the attention-check result.
2. Reports descriptives per condition.
3. CHOOSES the statistical test from the data rather than assuming one:

     * 2 conditions present:
         - normal residuals + equal variances -> independent t-test
         - normal residuals + unequal variances -> Welch's t-test
         - non-normal                          -> Mann-Whitney U
     * 3+ conditions:
         - normal residuals + equal variances  -> one-way ANOVA
         - normal residuals + unequal variances-> Welch's ANOVA
         - non-normal                          -> Kruskal-Wallis

   Normality is tested on the residuals (Shapiro-Wilk); variance homogeneity
   with Levene's test. Small groups (n < 3) are reported but not tested.
4. Corrects for testing 7 outcomes (6 subscales + overall) with
   Benjamini-Hochberg FDR, and prints which results are significant.
5. Runs pairwise post-hoc tests (with correction) for any significant omnibus.
6. Reports an effect size for every test (Hedges' g, eta-squared, or
   epsilon-squared / rank-biserial as appropriate).

USAGE
-----
        python3 analyze_tlx.py AI_comparisons_participants \
            --plots AI_comparisons_tlx_figures

Input files may be either era (pre- or post-fix); only the surveys are read.

SCORING NOTE: the performance scale is anchored Good (low) -> Poor (high), the
standard NASA-TLX direction, so a LOWER score = BETTER self-rated performance.
It is already oriented like the other five subscales (low = good, high = costly),
so no reverse-coding is applied and overallWorkload is their plain mean.

Requires: pandas, scipy, numpy.
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # file output; no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exclusions  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

SUBSCALES = [
    "mentalDemand", "physicalDemand", "temporalDemand",
    "performance", "effort", "frustration",
]
OUTCOMES = SUBSCALES + ["overallWorkload"]

CONDITION_ORDER = ["no_ai", "peaks_only", "confidence", "bars_only", "threshold_bars"]


# ── Loading ────────────────────────────────────────────────────────────────
def load_participants(folder, args):
    """Load every submission, score the TLX, and evaluate exclusions transparently."""
    records = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skipping {os.path.basename(path)}: {exc}", file=sys.stderr)
            continue

        data = doc.get("data", doc)
        if not isinstance(data, dict):
            continue

        fname = os.path.basename(path)
        user = data.get("userName") or doc.get("userName") or fname
        row = {"file": fname, "userName": user,
               "condition": data.get("visualizationMode")}

        tlx = (data.get("surveys") or {}).get("nasaTLX") or {}
        subs = tlx.get("subscaleScores") or {}
        for name, val in subs.items():
            if isinstance(val, dict) and not val.get("isAttentionCheck"):
                row[name] = val.get("score")

        overall = tlx.get("overallWorkload")
        if overall is None:
            vals = [row.get(s) for s in SUBSCALES if row.get(s) is not None]
            overall = float(np.mean(vals)) if vals else None
        row["overallWorkload"] = overall

        n_chroms = len(data.get("chromatograms") or [])
        ev = exclusions.evaluate(data, n_chroms, args)

        records.append(dict(
            key=(data.get("prolificPid") or user or fname),
            fname=fname, userName=user, condition=data.get("visualizationMode"),
            dur=data.get("sessionDurationMs"), row=row, has_tlx=bool(subs), **ev,
        ))

    if not args.keep_duplicates:
        kept, dropped = exclusions.resolve_duplicates(records)
        records = kept + dropped

    exclusions.print_audit(records, args)

    included = [r for r in records if not r["excluded"]]
    rows = [r["row"] for r in included if r["has_tlx"]]

    n_no_tlx = sum(1 for r in included if not r["has_tlx"])
    if n_no_tlx:
        print(f"\nNote: {n_no_tlx} included participant(s) have no NASA-TLX responses "
              f"and cannot contribute to this analysis.")

    return pd.DataFrame(rows)


# ── Test selection + effect sizes ──────────────────────────────────────────
def hedges_g(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1 - (3 / (4 * (na + nb) - 9))  # small-sample correction
    return d * J


def rank_biserial(a, b, u):
    return (2 * u) / (len(a) * len(b)) - 1


def welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    w = n / v
    sw = w.sum()
    mbar = (w * m).sum() / sw
    num = ((w * (m - mbar) ** 2).sum()) / (k - 1)
    lam = (((1 - w / sw) ** 2) / (n - 1)).sum()
    den = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lam
    F = num / den
    df1 = k - 1
    df2 = 1 / ((3 / (k ** 2 - 1)) * lam)
    p = stats.f.sf(F, df1, df2)
    return F, p, df1, df2


def choose_and_run(groups, labels):
    """Pick the appropriate omnibus/2-group test from the data itself."""
    usable = [(g, l) for g, l in zip(groups, labels) if len(g) >= 3]
    if len(usable) < 2:
        return {"test": "not run (need >=2 groups with n>=3)", "p": np.nan,
                "stat": np.nan, "effect": np.nan, "effect_name": "-", "notes": ""}

    gs = [np.asarray(g, float) for g, _ in usable]
    ls = [l for _, l in usable]

    # Normality of residuals (within-group centered), Shapiro-Wilk.
    resid = np.concatenate([g - g.mean() for g in gs])
    if 3 <= len(resid) <= 5000:
        p_norm = stats.shapiro(resid).pvalue
    else:
        p_norm = 1.0
    normal = p_norm > 0.05

    # Homogeneity of variance (Levene, median-centered = Brown-Forsythe).
    p_lev = stats.levene(*gs, center="median").pvalue if len(gs) >= 2 else 1.0
    equal_var = p_lev > 0.05

    notes = f"Shapiro p={p_norm:.3f} ({'normal' if normal else 'non-normal'}); " \
            f"Levene p={p_lev:.3f} ({'equal var' if equal_var else 'unequal var'})"

    if len(gs) == 2:
        a, b = gs
        if normal and equal_var:
            st, p = stats.ttest_ind(a, b, equal_var=True)
            return dict(test="independent t-test", stat=st, p=p,
                        effect=hedges_g(a, b), effect_name="Hedges g",
                        notes=notes, groups=ls)
        if normal and not equal_var:
            st, p = stats.ttest_ind(a, b, equal_var=False)
            return dict(test="Welch's t-test", stat=st, p=p,
                        effect=hedges_g(a, b), effect_name="Hedges g",
                        notes=notes, groups=ls)
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return dict(test="Mann-Whitney U", stat=u, p=p,
                    effect=rank_biserial(a, b, u), effect_name="rank-biserial",
                    notes=notes, groups=ls)

    # 3+ groups
    if normal and equal_var:
        F, p = stats.f_oneway(*gs)
        grand = np.concatenate(gs)
        ss_b = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in gs)
        ss_t = ((grand - grand.mean()) ** 2).sum()
        eta2 = ss_b / ss_t if ss_t else np.nan
        return dict(test="one-way ANOVA", stat=F, p=p,
                    effect=eta2, effect_name="eta^2", notes=notes, groups=ls)
    if normal and not equal_var:
        F, p, df1, df2 = welch_anova(gs)
        return dict(test="Welch's ANOVA", stat=F, p=p,
                    effect=np.nan, effect_name="-",
                    notes=notes + f"; df=({df1:.0f},{df2:.1f})", groups=ls)

    H, p = stats.kruskal(*gs)
    n = sum(len(g) for g in gs)
    k = len(gs)
    eps2 = (H - k + 1) / (n - k) if n > k else np.nan
    return dict(test="Kruskal-Wallis", stat=H, p=p,
                effect=eps2, effect_name="epsilon^2", notes=notes, groups=ls)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    adj = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return adj
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = min(prev, p[i] * m / k)
        adj[i] = prev = val
    return adj


def posthoc(groups, labels, parametric):
    out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = np.asarray(groups[i], float), np.asarray(groups[j], float)
            if len(a) < 3 or len(b) < 3:
                continue
            if parametric:
                st, p = stats.ttest_ind(a, b, equal_var=False)  # Welch pairwise
                eff, name, test = hedges_g(a, b), "Hedges g", "Welch t"
            else:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                st, eff, name, test = u, rank_biserial(a, b, u), "rank-biserial", "Mann-Whitney"
            out.append(dict(pair=f"{labels[i]} vs {labels[j]}", test=test,
                            stat=st, p=p, effect=eff, effect_name=name,
                            n1=len(a), n2=len(b),
                            mean1=a.mean(), mean2=b.mean()))
    if out:
        adj = bh_fdr([o["p"] for o in out])
        for o, a in zip(out, adj):
            o["p_adj"] = a
    return out


# ── Plots ──────────────────────────────────────────────────────────────────
# Colorblind-safe palette (same orange/blue family as the study interface).
COND_COLORS = {
    "no_ai":          "#fc8d59",
    "peaks_only":     "#fdd0a2",
    "confidence":     "#91bfdb",
    "bars_only":      "#4575b4",
    "threshold_bars": "#1a4a89",
}
DEFAULT_COLORS = ["#fc8d59", "#fdd0a2", "#91bfdb", "#4575b4", "#999999"]

PRETTY = {
    "mentalDemand": "Mental Demand",
    "physicalDemand": "Physical Demand",
    "temporalDemand": "Temporal Demand",
    "performance": "Performance\n(lower = better)",
    "effort": "Effort",
    "frustration": "Frustration",
    "overallWorkload": "Overall Workload",
}


def _sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _draw_box(ax, df, outcome, conds, result, alpha, annotate=True):
    """One boxplot: conditions on x, scores on y, raw points overlaid."""
    groups = [df.loc[df["condition"] == c, outcome].dropna().values for c in conds]
    colors = [COND_COLORS.get(c, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
              for i, c in enumerate(conds)]

    bp = ax.boxplot(
        [g if len(g) else [np.nan] for g in groups],
        patch_artist=True, widths=0.55, showfliers=False,
        medianprops=dict(color="#1e293b", linewidth=2),
        whiskerprops=dict(color="#64748b"),
        capprops=dict(color="#64748b"),
        boxprops=dict(edgecolor="#475569", linewidth=1),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Individual participants (jittered) — essential to see with small n.
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups, start=1):
        if not len(g):
            continue
        x = rng.normal(i, 0.055, size=len(g))
        ax.scatter(x, g, s=26, color="#1e293b", alpha=0.65,
                   zorder=3, edgecolors="white", linewidths=0.6)
        # mean marker
        ax.scatter([i], [np.mean(g)], marker="D", s=42, color="#b30000",
                   zorder=4, edgecolors="white", linewidths=0.8)

    ax.set_xticks(range(1, len(conds) + 1))
    ax.set_xticklabels([f"{c}\n(n={len(g)})" for c, g in zip(conds, groups)],
                       fontsize=8)
    ax.set_ylabel("score (0–100)", fontsize=9)
    ax.set_ylim(-5, 118)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    title = PRETTY.get(outcome, outcome)
    if result and not np.isnan(result.get("p_adj", np.nan)):
        sig = result["p_adj"] < alpha
        eff = result.get("effect")
        eff_txt = ""
        if eff is not None and not (isinstance(eff, float) and np.isnan(eff)):
            eff_txt = f", {result['effect_name']}={eff:.2f}"
        title += (f"\n{result['test']}: p_FDR={result['p_adj']:.3f}{eff_txt}"
                  f"{'  ✱ significant' if sig else ''}")
    ax.set_title(title, fontsize=9.5,
                 fontweight="bold" if (result and not np.isnan(result.get("p_adj", np.nan))
                                       and result["p_adj"] < alpha) else "normal")

    # Bracket significant pairwise comparisons.
    if annotate and result and len(conds) > 2 and \
       not np.isnan(result.get("p_adj", np.nan)) and result["p_adj"] < alpha:
        parametric = "ANOVA" in result["test"] or "t-test" in result["test"]
        ph = [o for o in posthoc(groups, conds, parametric) if o["p_adj"] < alpha]
        y = 100.0
        for o in ph[:6]:
            a, b = o["pair"].split(" vs ")
            i, j = conds.index(a) + 1, conds.index(b) + 1
            ax.plot([i, i, j, j], [y, y + 3, y + 3, y], lw=1.0, c="#475569")
            ax.text((i + j) / 2, y + 3.2, _sig_stars(o["p_adj"]),
                    ha="center", va="bottom", fontsize=8.5, color="#1e293b")
            y += 8
        ax.set_ylim(-5, max(118, y + 10))


def make_plots(df, conds, results, outdir, alpha):
    os.makedirs(outdir, exist_ok=True)
    res_by_outcome = {r["outcome"]: r for r in results}

    # 1) Overview grid: every outcome on one figure.
    ncols = 4
    nrows = int(np.ceil(len(OUTCOMES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, outcome in zip(axes, OUTCOMES):
        _draw_box(ax, df, outcome, conds, res_by_outcome.get(outcome), alpha)
    for ax in axes[len(OUTCOMES):]:
        ax.axis("off")
    fig.suptitle("NASA-TLX by condition  (box = IQR, line = median, "
                 "red ◆ = mean, dots = participants)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = os.path.join(outdir, "nasa_tlx_all_outcomes.png")
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    # 2) One figure per outcome.
    for outcome in OUTCOMES:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        _draw_box(ax, df, outcome, conds, res_by_outcome.get(outcome), alpha)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"nasa_tlx_{outcome}.png"), dpi=150)
        plt.close(fig)

    # 3) All six subscales side by side, grouped by condition.
    fig, ax = plt.subplots(figsize=(12, 5.4))
    width = 0.8 / max(len(conds), 1)
    rng = np.random.default_rng(1)
    for ci, c in enumerate(conds):
        color = COND_COLORS.get(c, DEFAULT_COLORS[ci % len(DEFAULT_COLORS)])
        positions = [si + (ci - (len(conds) - 1) / 2) * width
                     for si in range(len(SUBSCALES))]
        data = [df.loc[df["condition"] == c, s].dropna().values for s in SUBSCALES]
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data],
                        positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#1e293b", linewidth=1.5),
                        boxprops=dict(edgecolor="#475569", linewidth=0.8),
                        whiskerprops=dict(color="#94a3b8"),
                        capprops=dict(color="#94a3b8"))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for pos, d in zip(positions, data):
            if len(d):
                ax.scatter(rng.normal(pos, width * 0.06, len(d)), d, s=12,
                           color="#1e293b", alpha=0.5, zorder=3)
        ax.plot([], [], color=color, lw=8, alpha=0.75, label=c)

    ax.set_xticks(range(len(SUBSCALES)))
    ax.set_xticklabels([PRETTY.get(s, s).replace("\n", " ") for s in SUBSCALES],
                       fontsize=9)
    ax.set_ylabel("score (0–100)")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(title="condition", fontsize=9, ncol=len(conds), loc="upper center",
              bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_title("NASA-TLX subscales by condition", fontsize=12,
                 fontweight="bold", pad=34)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "nasa_tlx_subscales_grouped.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots written to {outdir}/")
    print(f"  nasa_tlx_all_outcomes.png       (overview grid)")
    print(f"  nasa_tlx_subscales_grouped.png  (all subscales, grouped)")
    print(f"  nasa_tlx_<outcome>.png          (one per outcome)")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--alpha", type=float, default=0.05)
    exclusions.add_exclusion_args(ap)
    ap.add_argument("--csv", help="write the per-participant table to this CSV")
    ap.add_argument("--plots", nargs="?", const="plots", default=None,
                    metavar="DIR",
                    help="save boxplots (default dir: ./plots)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip plotting entirely")
    args = ap.parse_args()

    df = load_participants(args.folder, args)
    if df.empty:
        print("\nNo participants with NASA-TLX data remain after exclusions.")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("NASA-TLX ANALYSIS")
    print("=" * 78)
    print(f"Analyzing {len(df)} participant(s) with NASA-TLX responses.")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Per-participant table written to {args.csv}")

    conds = [c for c in CONDITION_ORDER if c in set(df["condition"].dropna())]
    conds += [c for c in sorted(set(df["condition"].dropna())) if c not in conds]

    print("\nParticipants per condition:")
    for c in conds:
        print(f"  {c:16} n = {(df['condition'] == c).sum()}")
    if len(conds) < 2:
        print("\nOnly one condition present — no between-condition tests possible.")
        sys.exit(0)

    # Descriptives
    print("\n" + "-" * 78)
    print("DESCRIPTIVES  (mean ± SD [median])")
    print("-" * 78)
    hdr = f"{'outcome':18}" + "".join(f"{c:>20}" for c in conds)
    print(hdr)
    for out in OUTCOMES:
        line = f"{out:18}"
        for c in conds:
            v = df.loc[df["condition"] == c, out].dropna().values
            line += f"{(f'{v.mean():.1f}±{v.std(ddof=1):.1f} [{np.median(v):.0f}]' if len(v) > 1 else (f'{v[0]:.1f}' if len(v) == 1 else '-')):>20}"
        print(line)

    # Omnibus tests, one per outcome
    print("\n" + "-" * 78)
    print("OMNIBUS TESTS  (test chosen from normality + variance homogeneity)")
    print("-" * 78)
    results = []
    for out in OUTCOMES:
        groups = [df.loc[df["condition"] == c, out].dropna().values for c in conds]
        r = choose_and_run(groups, conds)
        r["outcome"] = out
        r["_groups"] = groups
        results.append(r)

    adj = bh_fdr([r["p"] for r in results])
    for r, a in zip(results, adj):
        r["p_adj"] = a

    for r in results:
        p = r["p"]
        pa = r["p_adj"]
        star = "  ***SIGNIFICANT***" if (not np.isnan(pa) and pa < args.alpha) else ""
        eff = "" if (r.get("effect") is None or (isinstance(r["effect"], float) and np.isnan(r["effect"]))) \
              else f", {r['effect_name']}={r['effect']:.3f}"
        pstr = "n/a" if np.isnan(p) else f"{p:.4f}"
        pastr = "n/a" if np.isnan(pa) else f"{pa:.4f}"
        print(f"\n{r['outcome']}")
        print(f"   test: {r['test']}")
        if r.get("notes"):
            print(f"   assumptions: {r['notes']}")
        stat = r.get("stat")
        statstr = "n/a" if stat is None or (isinstance(stat, float) and np.isnan(stat)) else f"{stat:.3f}"
        print(f"   stat={statstr}, p={pstr}, p_FDR={pastr}{eff}{star}")

    # Significant summary + post-hoc
    sig = [r for r in results if not np.isnan(r["p_adj"]) and r["p_adj"] < args.alpha]

    print("\n" + "=" * 78)
    print(f"SIGNIFICANT RESULTS (FDR-corrected across {len(OUTCOMES)} outcomes, alpha={args.alpha})")
    print("=" * 78)
    if not sig:
        print("None. No outcome differed significantly between conditions after correction.")
    for r in sig:
        print(f"\n* {r['outcome']}: {r['test']}, p_FDR={r['p_adj']:.4f}"
              + (f", {r['effect_name']}={r['effect']:.3f}" if not (isinstance(r['effect'], float) and np.isnan(r['effect'])) else ""))
        for c in conds:
            v = df.loc[df["condition"] == c, r["outcome"]].dropna().values
            if len(v):
                print(f"    {c:16} mean={v.mean():6.1f}  n={len(v)}")
        if len(conds) > 2:
            parametric = "ANOVA" in r["test"] or "t-test" in r["test"]
            ph = posthoc(r["_groups"], conds, parametric)
            if ph:
                print("    post-hoc (BH-corrected):")
                for o in ph:
                    mark = " *" if o["p_adj"] < args.alpha else ""
                    print(f"      {o['pair']:34} {o['test']:12} "
                          f"p={o['p']:.4f} p_adj={o['p_adj']:.4f} "
                          f"{o['effect_name']}={o['effect']:.2f}{mark}")

    # Boxplots (on by default; use --no-plots to skip)
    if not args.no_plots:
        outdir = args.plots or "plots"
        try:
            make_plots(df, conds, results, outdir, args.alpha)
        except Exception as exc:  # noqa: BLE001
            print(f"\n! plotting failed: {exc}")

    print("\nNote: the app's performance scale is anchored Good (low) -> Poor (high), "
          "the standard NASA-TLX direction, so a LOWER score means BETTER self-rated "
          "performance. It is therefore already oriented like the other subscales "
          "(low = good, high = demanding) and needs no reverse-coding: "
          "overallWorkload is the plain mean of the six subscales.")


if __name__ == "__main__":
    main()