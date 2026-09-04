#!/usr/bin/env python3
"""
Statistical tests on peak-annotation accuracy across the AI conditions.

UNIT OF ANALYSIS: MACRO (per participant), not micro.
-----------------------------------------------------
Each participant contributes ONE number per outcome: the mean of their
per-chromatogram scores. Tests are run across those participant-level values.

Why macro and not micro (pooling TP/FP/FN counts across trials/people):

  1. INDEPENDENCE. Statistical tests assume independent observations. Micro
     pooling treats every peak as an observation, but peaks within a participant
     are correlated (a careless participant is sloppy on all 32 charts). Pooling
     inflates n from ~dozens of people to ~thousands of peaks and produces
     anti-conservative p-values -- near-guaranteed "significance" that is an
     artifact. The participant is the randomized unit, so the participant must
     be the unit of analysis.
  2. EQUAL WEIGHTING. Micro weights peak-rich chromatograms (and participants who
     annotated more charts) more heavily. Since participants may quit early,
     micro would let the most persistent people dominate the condition mean.
  3. CONVENTION. Per-subject aggregation then between-subject tests is the
     standard design for a between-subjects HCI experiment like this.

Micro figures are still printed as a descriptive summary -- they are useful for
reporting overall system performance -- but they are NOT what the tests use.

OUTCOMES TESTED (per participant): f1, precision, recall, mean_iou,
mean_apex_error, plus FP and FN counts per chromatogram.

TEST SELECTION is data-driven (same logic as the NASA-TLX script):
  2 groups : t-test / Welch's t-test / Mann-Whitney U
  3+ groups: one-way ANOVA / Welch's ANOVA / Kruskal-Wallis
chosen via Shapiro-Wilk (normality of residuals) and Levene (equal variance),
with Benjamini-Hochberg FDR correction across outcomes and BH-corrected
pairwise post-hoc tests for significant omnibus results.

USAGE
-----
    python3 analyze_accuracy.py --submissions AI_comparisons_participants \
        --ai-detections DATA_DIR/synthetic_data --ground-truth DATA_DIR/synthetic_data \
            --plots AI_comparisons_acc_figures

Requires: numpy, pandas, scipy, matplotlib. Must sit next to score_peaks.py.
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the (unmodified) matching + loading logic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_peaks import (  # noqa: E402
    compute_metrics, load_ground_truth, chrom_stem,
    submitted_triples, iter_submissions, prf, was_visited,
)
import exclusions  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Outcomes tested per participant: (column, plot title, higher_is_better)
OUTCOMES = [
    ("f1", "F1", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("mean_iou", "Mean IoU", True),
    ("mean_apex_error", "Mean apex error\n(lower = better)", False),
    ("fp_per_chrom", "False positives\nper chromatogram", False),
    ("fn_per_chrom", "False negatives\nper chromatogram", False),
]

CONDITION_ORDER = ["no_ai", "peaks_only", "confidence", "bars_only", "threshold_bars"]

# Colorblind-safe palette (same orange/blue family as the study interface).
COND_COLORS = {
    "no_ai": "#fc8d59",
    "peaks_only": "#fdd0a2",
    "confidence": "#91bfdb",
    "bars_only": "#4575b4",
    "threshold_bars": "#1a4a89",
}

# ── Statistics (test chosen from the data) ─────────────────────────────────
def hedges_g(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return ((np.mean(a) - np.mean(b)) / sp) * (1 - (3 / (4 * (na + nb) - 9)))


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
    F = num / (1 + (2 * (k - 2) / (k ** 2 - 1)) * lam)
    df2 = 1 / ((3 / (k ** 2 - 1)) * lam)
    return F, stats.f.sf(F, k - 1, df2), k - 1, df2


def choose_and_run(groups, labels):
    usable = [(np.asarray(g, float), l) for g, l in zip(groups, labels) if len(g) >= 3]
    if len(usable) < 2:
        return dict(test="not run (need >=2 groups with n>=3)", stat=np.nan, p=np.nan,
                    effect=np.nan, effect_name="-", notes="")
    gs = [g for g, _ in usable]
    ls = [l for _, l in usable]

    resid = np.concatenate([g - g.mean() for g in gs])
    p_norm = stats.shapiro(resid).pvalue if 3 <= len(resid) <= 5000 else 1.0
    normal = p_norm > 0.05
    p_lev = stats.levene(*gs, center="median").pvalue
    equal_var = p_lev > 0.05
    notes = (f"Shapiro p={p_norm:.3f} ({'normal' if normal else 'non-normal'}); "
             f"Levene p={p_lev:.3f} ({'equal var' if equal_var else 'unequal var'})")

    if len(gs) == 2:
        a, b = gs
        if normal:
            st, p = stats.ttest_ind(a, b, equal_var=equal_var)
            return dict(test="independent t-test" if equal_var else "Welch's t-test",
                        stat=st, p=p, effect=hedges_g(a, b), effect_name="Hedges g",
                        notes=notes, groups=ls)
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return dict(test="Mann-Whitney U", stat=u, p=p, effect=rank_biserial(a, b, u),
                    effect_name="rank-biserial", notes=notes, groups=ls)

    if normal and equal_var:
        F, p = stats.f_oneway(*gs)
        grand = np.concatenate(gs)
        ss_b = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in gs)
        ss_t = ((grand - grand.mean()) ** 2).sum()
        return dict(test="one-way ANOVA", stat=F, p=p,
                    effect=(ss_b / ss_t if ss_t else np.nan), effect_name="eta^2",
                    notes=notes, groups=ls)
    if normal and not equal_var:
        F, p, df1, df2 = welch_anova(gs)
        return dict(test="Welch's ANOVA", stat=F, p=p, effect=np.nan, effect_name="-",
                    notes=notes + f"; df=({df1:.0f},{df2:.1f})", groups=ls)

    H, p = stats.kruskal(*gs)
    n = sum(len(g) for g in gs)
    k = len(gs)
    return dict(test="Kruskal-Wallis", stat=H, p=p,
                effect=((H - k + 1) / (n - k) if n > k else np.nan),
                effect_name="epsilon^2", notes=notes, groups=ls)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    adj = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if not len(idx):
        return adj
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        adj[i] = prev = min(prev, p[i] * m / (m - rank + 1))
    return adj


def posthoc(groups, labels, parametric):
    out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = np.asarray(groups[i], float), np.asarray(groups[j], float)
            if len(a) < 3 or len(b) < 3:
                continue
            if parametric:
                st, p = stats.ttest_ind(a, b, equal_var=False)
                eff, name, test = hedges_g(a, b), "Hedges g", "Welch t"
            else:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                st, eff, name, test = u, rank_biserial(a, b, u), "rank-biserial", "Mann-Whitney"
            out.append(dict(pair=f"{labels[i]} vs {labels[j]}", test=test, p=p,
                            effect=eff, effect_name=name,
                            mean1=a.mean(), mean2=b.mean(), n1=len(a), n2=len(b)))
    if out:
        for o, a in zip(out, bh_fdr([o["p"] for o in out])):
            o["p_adj"] = a
    return out


# ── Build the participant-level (macro) table ──────────────────────────────
def build_tables(submissions, ground_truth, args):
    gt_index = load_ground_truth(ground_truth)
    if not gt_index:
        print(f"No ground truth found under {ground_truth}")
        sys.exit(1)

    # Pass 1: read every submission, score its chromatograms, evaluate exclusions.
    records = []
    n_unvisited_skipped = {}
    for fname, doc, data in iter_submissions(submissions):
        user = data.get("userName") or doc.get("userName") or fname
        cond = data.get("visualizationMode")

        rows = []
        for ch in data.get("chromatograms") or []:
            # Skip charts the participant never opened. In the AI conditions these
            # are still pre-filled with the AI's peaks, so scoring them would
            # measure the AI detector rather than the person (see was_visited).
            if not args.include_unvisited and not was_visited(ch):
                n_unvisited_skipped[cond] = n_unvisited_skipped.get(cond, 0) + 1
                continue
            stem = chrom_stem(ch.get("file"))
            gt = gt_index.get(stem)
            if gt is None:
                continue
            sub = submitted_triples(ch.get("annotations"),
                                    user_edited_only=args.user_edited_only,
                                    exclude_untouched_ai=args.exclude_untouched_ai)
            m = compute_metrics(sub, gt)
            apex_errs = [abs(sub[i][1] - gt[j][1]) for i, j, _ in m["matched_pairs"]]
            rows.append(dict(
                userName=user, condition=cond, chromatogram=stem,
                chromType=stem.split("_")[-1] if "_" in stem else "",
                TP=m["TP"], FP=m["FP"], FN=m["FN"],
                precision=m["precision"], recall=m["recall"], f1=m["f1"],
                mean_iou=m["mean_iou"],
                mean_apex_error=float(np.mean(apex_errs)) if apex_errs else np.nan,
            ))

        ev = exclusions.evaluate(data, len(rows), args)
        records.append(dict(
            key=(data.get("prolificPid") or user or fname),
            fname=fname, userName=user, condition=cond,
            dur=data.get("sessionDurationMs"), rows=rows, **ev,
        ))

    # Duplicates: keep the most complete submission per person.
    if not args.keep_duplicates:
        kept, dup_dropped = exclusions.resolve_duplicates(records)
        records = kept + dup_dropped   # keep all rows for the audit; dups marked excluded

    exclusions.print_audit(records, args)

    if n_unvisited_skipped and not args.include_unvisited:
        total_skipped = sum(n_unvisited_skipped.values())
        print(f"\nSkipped {total_skipped} chromatogram(s) the participant never opened "
              f"(visitCount=0). Only charts they actually worked on are scored.")
        for c, n in sorted(n_unvisited_skipped.items(), key=lambda kv: str(kv[0])):
            print(f"   {str(c):16} {n:>5} unvisited charts skipped")
        print("   (In the AI conditions unvisited charts still carry the AI's suggested")
        print("    peaks, so scoring them would measure the AI, not the participant.)")

    included = [r for r in records if not r["excluded"]]

    chrom_rows = [row for r in included for row in r["rows"]]
    if not chrom_rows:
        print("\nNo chromatograms left to score after exclusions.")
        sys.exit(1)

    chroms = pd.DataFrame(chrom_rows)

    # MACRO: average each participant's per-chromatogram scores -> one row/person.
    parts = []
    for (user, cond), g in chroms.groupby(["userName", "condition"], dropna=False):
        parts.append(dict(
            userName=user, condition=cond, n_chromatograms=len(g),
            f1=g["f1"].mean(),
            precision=g["precision"].mean(),
            recall=g["recall"].mean(),
            mean_iou=g["mean_iou"].mean(),
            mean_apex_error=g["mean_apex_error"].mean(skipna=True),
            fp_per_chrom=g["FP"].mean(),
            fn_per_chrom=g["FN"].mean(),
            micro_f1=prf(g["TP"].sum(), g["FP"].sum(), g["FN"].sum())[2],
            TP=int(g["TP"].sum()), FP=int(g["FP"].sum()), FN=int(g["FN"].sum()),
        ))
    return chroms, pd.DataFrame(parts)


# ── Plots ──────────────────────────────────────────────────────────────────
def score_ai_per_chrom(ai_root, gt_index):
    """
    Score the raw AI detector on EVERY chromatogram that has a detection CSV and
    a ground-truth entry, using the same matcher as the humans.

    Returns dict: stem -> {f1, precision, recall, mean_iou, mean_apex_error,
    fp_per_chrom, fn_per_chrom}. The AI's score on a chart is a fixed property of
    that chart, so this table is computed once and then aggregated whatever way
    the caller wants (all charts, or matched to each participant's visited set).
    """
    import csv
    import glob as _glob

    # Accepted column names (normalized: lowercased, stripped, BOM removed).
    START_KEYS = ("start_time", "start", "start_min", "start_x", "left_time", "start_rt")
    APEX_KEYS = ("apex_time", "apex", "apex_min", "apex_x", "rt", "retention_time", "peak_time")
    END_KEYS = ("end_time", "end", "end_min", "end_x", "right_time", "end_rt")

    def norm(k):
        return (k or "").replace("\ufeff", "").strip().lower()

    def pick(row_norm, candidates):
        for c in candidates:
            if c in row_norm and row_norm[c] not in ("", None):
                return row_norm[c]
        return None

    out = {}
    n_files = 0
    all_csvs = _glob.glob(os.path.join(ai_root, "**", "*.csv"), recursive=True)
    # Only the detection peak tables — NOT the raw-signal CSVs that live in the
    # sibling chromatograms/ folders and share the same chromatogram stem. Reading
    # those would parse 0 peaks and (depending on filesystem order) clobber the
    # real score with zeros. Fall back to all CSVs only if none are named *_peak_table.
    peak_csvs = [p for p in all_csvs if "peak_table" in os.path.basename(p).lower()]
    use_csvs = peak_csvs if peak_csvs else all_csvs
    for path in use_csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        stem = name.replace("_peak_table", "").replace("_detections", "")
        gt = gt_index.get(stem)
        if gt is None:
            continue
        n_files += 1
        try:
            raw = list(csv.DictReader(open(path, newline="")))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! unreadable AI CSV {path}: {exc}", file=sys.stderr)
            continue

        triples, n_bad = [], 0
        header_cols = set()
        for r in raw:
            rn = {norm(k): v for k, v in r.items()}
            header_cols = set(rn.keys())
            sv, xv, ev = pick(rn, START_KEYS), pick(rn, APEX_KEYS), pick(rn, END_KEYS)
            try:
                s, x, e = float(sv), float(xv), float(ev)
            except (TypeError, ValueError):
                n_bad += 1
                continue
            if e < s:
                s, e = e, s
            triples.append((s, x, e))

        # Loud warning when a CSV yields no usable peaks -- the exact condition
        # that silently produced 0 recall/precision before.
        if not triples:
            print(f"  ! {os.path.basename(path)}: parsed 0 peaks from {len(raw)} row(s). "
                  f"Columns present: {sorted(header_cols)}. "
                  f"Expected start/apex/end time columns "
                  f"(any of start_time/apex_time/end_time). This chart would score "
                  f"recall=0 -- fix the column names or tell me what they are.",
                  file=sys.stderr)
            continue
        if n_bad:
            print(f"  ! {os.path.basename(path)}: skipped {n_bad} unparseable row(s).",
                  file=sys.stderr)

        triples.sort(key=lambda t: t[1])
        m = compute_metrics(triples, gt)
        apex = [abs(triples[i][1] - gt[j][1]) for i, j, _ in m["matched_pairs"]]
        out[stem] = dict(
            f1=m["f1"], precision=m["precision"], recall=m["recall"],
            mean_iou=m["mean_iou"],
            mean_apex_error=float(np.mean(apex)) if apex else np.nan,
            fp_per_chrom=float(m["FP"]), fn_per_chrom=float(m["FN"]),
        )
    return out


def _mean_over(ai_per_chrom, stems):
    """Mean AI score over a set of chromatogram stems (NaN-skipping)."""
    keys = [k for k, *_ in OUTCOMES]
    rows = [ai_per_chrom[s] for s in stems if s in ai_per_chrom]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean(skipna=True)) for k in keys if k in df}


def ai_baseline_all(ai_per_chrom):
    """AI over the full battery of charts — the stable detector benchmark."""
    return _mean_over(ai_per_chrom, list(ai_per_chrom.keys())), len(ai_per_chrom)


def ai_baseline_matched(ai_per_chrom, chroms):
    """
    AI aggregated the SAME way as the human macro mean: for each participant,
    average the AI's scores over exactly the charts THEY visited, then average
    across participants. Directly comparable to the human macro means.
    """
    keys = [k for k, *_ in OUTCOMES]
    per_participant = []
    for user, g in chroms.groupby("userName"):
        stems = list(g["chromatogram"].unique())
        mv = _mean_over(ai_per_chrom, stems)
        if mv:
            per_participant.append(mv)
    if not per_participant:
        return None, 0
    df = pd.DataFrame(per_participant)
    return {k: float(df[k].mean(skipna=True)) for k in keys if k in df}, len(per_participant)


def make_plots(parts, conds, results, outdir, alpha, ai_baseline=None):
    os.makedirs(outdir, exist_ok=True)
    res = {r["outcome"]: r for r in results}
    rng = np.random.default_rng(0)

    def draw(ax, key, title):
        groups = [parts.loc[parts["condition"] == c, key].dropna().values for c in conds]
        bp = ax.boxplot([g if len(g) else [np.nan] for g in groups], patch_artist=True,
                        widths=0.55, showfliers=False,
                        medianprops=dict(color="#1e293b", linewidth=2),
                        boxprops=dict(edgecolor="#475569"),
                        whiskerprops=dict(color="#64748b"), capprops=dict(color="#64748b"))
        for patch, c in zip(bp["boxes"], conds):
            patch.set_facecolor(COND_COLORS.get(c, "#999999"))
            patch.set_alpha(0.75)
        for i, g in enumerate(groups, start=1):
            if len(g):
                ax.scatter(rng.normal(i, 0.055, len(g)), g, s=26, color="#1e293b",
                           alpha=0.65, zorder=3, edgecolors="white", linewidths=0.6)
                ax.scatter([i], [np.mean(g)], marker="D", s=42, color="#b30000",
                           zorder=4, edgecolors="white", linewidths=0.8)
        # AI-only baseline (no human input): horizontal reference line.
        if ai_baseline and key in ai_baseline and ai_baseline[key] is not None \
                and not np.isnan(ai_baseline[key]):
            yb = ai_baseline[key]
            ax.axhline(yb, color="#6a1b9a", linestyle="--", linewidth=1.8, zorder=5)
            ax.text(len(conds) + 0.52, yb, f"AI only\n{yb:.2f}", color="#6a1b9a",
                    fontsize=7.5, va="center", ha="left", fontweight="bold")
        ax.set_xticks(range(1, len(conds) + 1))
        ax.set_xticklabels([f"{c}\n(n={len(g)})" for c, g in zip(conds, groups)], fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)

        r = res.get(key)
        t = title
        if r and not np.isnan(r.get("p_adj", np.nan)):
            sig = r["p_adj"] < alpha
            eff = r.get("effect")
            e = "" if eff is None or (isinstance(eff, float) and np.isnan(eff)) \
                else f", {r['effect_name']}={eff:.2f}"
            t += f"\n{r['test']}: p_FDR={r['p_adj']:.3f}{e}{'  ✱ significant' if sig else ''}"
            ax.set_title(t, fontsize=9.5, fontweight="bold" if sig else "normal")
            if sig and len(conds) > 2:
                parametric = "ANOVA" in r["test"] or "t-test" in r["test"]
                ph = [o for o in posthoc(groups, conds, parametric) if o["p_adj"] < alpha]
                lo, hi = ax.get_ylim()
                y = hi
                step = (hi - lo) * 0.07
                for o in ph[:6]:
                    a, b = o["pair"].split(" vs ")
                    i, j = conds.index(a) + 1, conds.index(b) + 1
                    ax.plot([i, i, j, j], [y, y + step * .35, y + step * .35, y], lw=1, c="#475569")
                    star = "***" if o["p_adj"] < .001 else "**" if o["p_adj"] < .01 else "*"
                    ax.text((i + j) / 2, y + step * .38, star, ha="center", fontsize=9)
                    y += step
                ax.set_ylim(lo, y + step)
        else:
            ax.set_title(t, fontsize=9.5)

    ncols = 4
    nrows = int(np.ceil(len(OUTCOMES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, title, _) in zip(axes, OUTCOMES):
        draw(ax, key, title)
    for ax in axes[len(OUTCOMES):]:
        ax.axis("off")
    fig.suptitle("Peak-annotation accuracy by condition — participant-level (macro) scores\n"
                 "box = IQR, line = median, red ◆ = mean, dots = participants",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, "peak_accuracy_all.png"), dpi=150)
    plt.close(fig)

    for key, title, _ in OUTCOMES:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        draw(ax, key, title)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"peak_{key}.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots written to {outdir}/")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--user-edited-only", action="store_true")
    ap.add_argument("--exclude-untouched-ai", action="store_true")
    ap.add_argument("--ai-detections", metavar="DIR",
                    help="root folder of AI detection CSVs (…/detections/*_peak_table.csv); "
                         "adds an 'AI only' baseline line to the plots and prints its scores")
    ap.add_argument("--ai-baseline-scope", choices=["all", "matched"], default="all",
                    help="which AI baseline the plot line uses: 'all' = AI over every "
                         "chart (stable benchmark, default); 'matched' = AI aggregated "
                         "like the human macro mean over each person's visited charts")
    ap.add_argument("--include-unvisited", action="store_true",
                    help="ALSO score charts the participant never opened "
                         "(NOT recommended: in AI conditions these are pre-filled "
                         "with AI peaks, which measures the AI, not the person)")

    exclusions.add_exclusion_args(ap)

    ap.add_argument("--csv", help="write the participant-level table here")
    ap.add_argument("--plots", nargs="?", const="plots", default=None, metavar="DIR")
    args = ap.parse_args()

    chroms, parts = build_tables(args.submissions, args.ground_truth, args)

    print("\n" + "=" * 88)
    print("PEAK-ANNOTATION ACCURACY BY CONDITION")
    print("=" * 88)
    print(f"{len(chroms)} chromatogram-trials from {len(parts)} analyzed participants.")
    print("\nUNIT OF ANALYSIS: macro (participant-level). Each participant contributes")
    print("one score per outcome = the mean of their per-chromatogram scores. Tests are")
    print("run across participants, because the participant is the randomized unit;")
    print("pooling peaks (micro) would violate independence and inflate significance.")

    if args.csv:
        parts.to_csv(args.csv, index=False)
        print(f"\nParticipant-level table -> {args.csv}")

    # AI-only baseline (optional). The AI's score on a chart is fixed, so we score
    # every chart once, then aggregate two ways:
    #   all     = the AI over the full battery of charts (stable benchmark)
    #   matched = the AI aggregated exactly like the human macro mean (per
    #             participant, over the charts THEY visited) — the fair comparison
    ai_baseline = None
    if args.ai_detections:
        gt_index = load_ground_truth(args.ground_truth)
        ai_per_chrom = score_ai_per_chrom(args.ai_detections, gt_index)
        print("\n" + "-" * 88)
        print("AI-ONLY BASELINE (raw detector, no human input)")
        print("-" * 88)
        if not ai_per_chrom:
            print(f"No AI detection CSVs matched the ground truth under {args.ai_detections}.")
        else:
            b_all, n_all = ai_baseline_all(ai_per_chrom)
            b_matched, n_p = ai_baseline_matched(ai_per_chrom, chroms)
            print(f"{'outcome':26}{'AI over ALL charts':>22}{'AI matched to visited':>24}")
            print(f"{'':26}{f'(n={n_all} charts)':>22}{f'(mean over {n_p} ppl)':>24}")
            for key, label, _ in OUTCOMES:
                a = b_all.get(key) if b_all else np.nan
                m = b_matched.get(key) if b_matched else np.nan
                a_s = f"{a:.3f}" if a is not None and not np.isnan(a) else "-"
                m_s = f"{m:.3f}" if m is not None and not np.isnan(m) else "-"
                print(f"{label.splitlines()[0]:26}{a_s:>22}{m_s:>24}")

            # Flag divergence: if all vs matched differ, the visited subset is
            # not representative of the full battery.
            if b_all and b_matched and not np.isnan(b_all.get("f1", np.nan)) \
                    and not np.isnan(b_matched.get("f1", np.nan)):
                gap = abs(b_all["f1"] - b_matched["f1"])
                if gap > 0.03:
                    print(f"\n! all-charts and matched AI F1 differ by {gap:.3f}: the charts "
                          f"participants visited are NOT a representative sample of the full "
                          f"battery. Prefer the MATCHED baseline for 'beat the AI' claims.")
                else:
                    print(f"\nall-charts and matched AI F1 agree to within {gap:.3f}: the "
                          f"visited charts are representative, so the choice barely matters.")

            scope = args.ai_baseline_scope
            ai_baseline = b_all if scope == "all" else b_matched
            print(f"\nPlot reference line uses the '{scope}' baseline "
                  f"(change with --ai-baseline-scope).")
            print("Boxes above the line = that human+AI condition beat the detector alone.")

    conds = [c for c in CONDITION_ORDER if c in set(parts["condition"].dropna())]
    conds += [c for c in sorted(set(parts["condition"].dropna())) if c not in conds]

    print("\n" + "-" * 88)
    print("DESCRIPTIVES — macro (mean ± SD across participants)")
    print("-" * 88)
    print(f"{'outcome':22}" + "".join(f"{c:>17}" for c in conds))
    for key, title, _ in OUTCOMES:
        line = f"{title.splitlines()[0]:22}"
        for c in conds:
            v = parts.loc[parts["condition"] == c, key].dropna().values
            line += f"{(f'{v.mean():.3f}±{v.std(ddof=1):.3f}' if len(v) > 1 else (f'{v[0]:.3f}' if len(v) else '-')):>17}"
        print(line)

    # Micro, descriptive only.
    print("\n" + "-" * 88)
    print("MICRO (counts pooled across all trials) — DESCRIPTIVE ONLY, not tested")
    print("-" * 88)
    print(f"{'condition':16}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>12}{'recall':>10}{'F1':>9}")
    for c in conds:
        g = chroms[chroms["condition"] == c]
        p, r, f = prf(g["TP"].sum(), g["FP"].sum(), g["FN"].sum())
        print(f"{c:16}{int(g['TP'].sum()):>7}{int(g['FP'].sum()):>7}{int(g['FN'].sum()):>7}"
              f"{p:>12.3f}{r:>10.3f}{f:>9.3f}")

    if len(conds) < 2:
        print("\nOnly one condition present — no tests possible.")
        sys.exit(0)

    print("\n" + "-" * 88)
    print("OMNIBUS TESTS (macro; test chosen from normality + variance homogeneity)")
    print("-" * 88)
    results = []
    for key, title, _ in OUTCOMES:
        groups = [parts.loc[parts["condition"] == c, key].dropna().values for c in conds]
        r = choose_and_run(groups, conds)
        r["outcome"] = key
        r["_groups"] = groups
        results.append(r)
    for r, a in zip(results, bh_fdr([r["p"] for r in results])):
        r["p_adj"] = a

    for r in results:
        pa = r["p_adj"]
        star = "  ***SIGNIFICANT***" if (not np.isnan(pa) and pa < args.alpha) else ""
        eff = r.get("effect")
        e = "" if eff is None or (isinstance(eff, float) and np.isnan(eff)) \
            else f", {r['effect_name']}={eff:.3f}"
        print(f"\n{r['outcome']}")
        print(f"   test: {r['test']}")
        if r.get("notes"):
            print(f"   assumptions: {r['notes']}")
        p_txt = "n/a" if np.isnan(r["p"]) else f"{r['p']:.4f}"
        pa_txt = "n/a" if np.isnan(pa) else f"{pa:.4f}"
        print(f"   p={p_txt}, p_FDR={pa_txt}{e}{star}")

    sig = [r for r in results if not np.isnan(r["p_adj"]) and r["p_adj"] < args.alpha]
    print("\n" + "=" * 88)
    print(f"SIGNIFICANT RESULTS (BH-FDR across {len(OUTCOMES)} outcomes, alpha={args.alpha})")
    print("=" * 88)
    if not sig:
        print("None survived correction.")
    for r in sig:
        eff = r.get("effect")
        e = "" if eff is None or (isinstance(eff, float) and np.isnan(eff)) \
            else f", {r['effect_name']}={eff:.3f}"
        print(f"\n* {r['outcome']}: {r['test']}, p_FDR={r['p_adj']:.4f}{e}")
        for c in conds:
            v = parts.loc[parts["condition"] == c, r["outcome"]].dropna().values
            if len(v):
                print(f"    {c:16} mean={v.mean():8.3f}  n={len(v)}")
        if len(conds) > 2:
            parametric = "ANOVA" in r["test"] or "t-test" in r["test"]
            for o in posthoc(r["_groups"], conds, parametric):
                mark = " *" if o["p_adj"] < args.alpha else ""
                print(f"      {o['pair']:34} {o['test']:12} p={o['p']:.4f} "
                      f"p_adj={o['p_adj']:.4f} {o['effect_name']}={o['effect']:.2f}{mark}")

    if args.plots is not None:
        make_plots(parts, conds, results, args.plots, args.alpha, ai_baseline)


if __name__ == "__main__":
    main()