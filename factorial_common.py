#!/usr/bin/env python3
"""
Shared machinery for the 2x2 (confidence x threshold) analyses.

THE DESIGN (no_ai was a separate study, so it is NOT included here):
    peaks_only      confidence=0, threshold=0   (baseline: suggestions only)
    bars_only       confidence=0, threshold=1   (+ threshold bars)
    confidence      confidence=1, threshold=0   (+ confidence)
    threshold_bars  confidence=1, threshold=1   (+ both)

Two model families, chosen to match how each outcome was measured:
  * TRIAL-LEVEL outcomes (accuracy, time) — one row per participant x chromatogram.
    Fit a linear MIXED model: outcome ~ confidence*threshold + trial_c, with a
    random intercept per participant and a variance component for chromatogram
    (crossed random effects). This preserves trial-level information and separates
    condition effects from participant ability and chromatogram difficulty.
  * PARTICIPANT-LEVEL outcomes (NASA-TLX, engagement) — measured ONCE per person,
    so there is nothing to nest. A mixed model would be mis-specified; instead fit
    the 2x2 factorial as an ordinary linear model: outcome ~ confidence*threshold.

Either way the coefficients answer the same three questions:
    confidence            does showing confidence help (when bars are absent)?
    threshold             do threshold bars help (when confidence is absent)?
    confidence:threshold  do the two together differ from the sum of their parts?

Requires: statsmodels, pandas, numpy, matplotlib.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

# condition -> (confidence, threshold)
FACTORS = {
    "peaks_only":     (0, 0),
    "bars_only":      (0, 1),
    "confidence":     (1, 0),
    "threshold_bars": (1, 1),
}
COND_ORDER = ["peaks_only", "bars_only", "confidence", "threshold_bars"]
COND_COLORS = {"peaks_only": "#fdd0a2", "bars_only": "#4575b4",
               "confidence": "#91bfdb", "threshold_bars": "#1a4a89"}
EFFECTS = ["confidence", "threshold", "confidence:threshold"]
EFFECT_LABEL = {
    "confidence": "Confidence\n(main effect)",
    "threshold": "Threshold bars\n(main effect)",
    "confidence:threshold": "Confidence × Threshold\n(interaction)",
    "trial_c": "Trial\n(fatigue/learning)",
}


def add_factors(df, condition_col="condition"):
    """Add numeric confidence/threshold columns; drop rows not in the 2x2."""
    df = df[df[condition_col].isin(FACTORS)].copy()
    df["confidence"] = df[condition_col].map(lambda c: FACTORS[c][0])
    df["threshold"] = df[condition_col].map(lambda c: FACTORS[c][1])
    return df


def trial_position_map(data):
    """
    Map chromatogram stem -> presentation position (1-based) from chromatogramOrder,
    which records each participant's RANDOMISED order. Falls back to array index if
    chromatogramOrder is absent. Because order is randomised per participant, trial
    and chromatogram are decorrelated, so the model can separate a position/fatigue
    effect from chromatogram difficulty.
    """
    order = data.get("chromatogramOrder")
    pos = {}
    if order:
        for e in order:
            base = e.get("baseName")
            if not base:
                name = e.get("name") or ""
                base = os.path.splitext(os.path.basename(name))[0]
            if base is not None and e.get("position") is not None:
                pos[base] = e["position"]
    return pos


EDIT_EVENT_SETS = {
    # what counts as an "edit". Pulled from data['editLog'] by event 'type'.
    "annotation": {"end_drag", "add_peak", "delete_peak"},  # actions that CHANGE an
                                                            # annotation (default, recommended)
    "interaction": None,                                    # all peak interactions EXCEPT
                                                            # panning (selects, badge-clicks,
                                                            # drags, adds, deletes, restores)
    "boundary": {"end_drag"},                               # boundary drags only
    "deletions": {"delete_peak"},                           # deletions only
    "all": None,                                            # every logged interaction incl. pan
}
# event types EXCLUDED for a given set (used when the 'keep' set is None)
EDIT_EVENT_EXCLUDE = {"interaction": {"pan"}}


def edit_counts(data, events="annotation"):
    """
    Per-chromatogram and total 'edit' counts from data['editLog'], filtered to the
    chosen event types (see EDIT_EVENT_SETS). Returns (per_chrom, total, from_log):
      per_chrom : dict {chromIdx(0-based) -> count}
      total     : int
      from_log  : True if computed from editLog; False if it fell back to the raw
                  per-chart editCount (which counts ALL interactions and can't be filtered).
    'annotation' = boundary drags + added + deleted peaks.
    'interaction' = every peak interaction except panning the view.
    """
    keep = EDIT_EVENT_SETS.get(events, EDIT_EVENT_SETS["annotation"])
    exclude = EDIT_EVENT_EXCLUDE.get(events, set())
    el = data.get("editLog")
    if el:
        per = {}
        for e in el:
            if not isinstance(e, dict):
                continue
            t = e.get("type")
            if t in exclude:
                continue
            if keep is None or t in keep:
                idx = e.get("chromIdx")
                per[idx] = per.get(idx, 0) + 1
        return per, sum(per.values()), True
    # fallback: raw all-event counts (cannot filter without the log)
    per = {i: (ch.get("editCount") or 0)
           for i, ch in enumerate(data.get("chromatograms") or [])}
    total = data.get("totalAnnotationEdits")
    if total is None:
        total = sum(per.values())
    return per, total, False


def fisher_z_diff(r1, n1, r2, n2, spearman=True):
    """
    Test whether two INDEPENDENT correlations differ (Fisher r-to-z). Returns (z, p),
    two-sided. Uses the 1.06 variance inflation for Spearman (Fieller), 1.0 for Pearson.
    """
    from scipy import stats as _st
    if (r1 is None or r2 is None or np.isnan(r1) or np.isnan(r2)
            or n1 is None or n2 is None or n1 <= 4 or n2 <= 4):
        return np.nan, np.nan
    z1 = np.arctanh(np.clip(r1, -0.999, 0.999))
    z2 = np.arctanh(np.clip(r2, -0.999, 0.999))
    fac = 1.06 if spearman else 1.0
    se = np.sqrt(fac / (n1 - 3) + fac / (n2 - 3))
    if se == 0:
        return np.nan, np.nan
    z = (z1 - z2) / se
    return float(z), float(2 * (1 - _st.norm.cdf(abs(z))))


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


def stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def fit_ols(df, outcome):
    d = df[[outcome, "confidence", "threshold"]].dropna()
    if len(d) < 8 or d[outcome].nunique() < 2:
        return None, d
    return smf.ols(f"Q('{outcome}') ~ confidence * threshold", data=d).fit(), d


def fit_mixed(df, outcome, trial=True, chrom_type=False, type_interactions=False):
    """
    Linear mixed model: outcome ~ confidence*threshold [+ trial_c] [+ chromatogram
    type terms], random intercept per participant + chromatogram variance component.
    Falls back to a participant-only random intercept if the crossed model fails.

    chrom_type=True adds chromatogram TYPE (control/drift/noise/tiny) as a fixed
    effect (control = reference), so difficulty becomes a testable coefficient
    rather than only a random component. type_interactions=True additionally adds
    confidence:type and threshold:type, testing whether the visualization effects
    differ by difficulty.
    """
    cols = [outcome, "confidence", "threshold", "participant_id", "chromatogram_id"]
    if trial:
        cols.append("trial_c")
    if chrom_type:
        cols.append("chrom_type")
    d = df[cols].dropna()
    if d[outcome].nunique() < 2 or d["participant_id"].nunique() < 4:
        return None, d, "insufficient data"
    formula = f"Q('{outcome}') ~ confidence * threshold" + (" + trial_c" if trial else "")
    if chrom_type and d["chrom_type"].nunique() > 1:
        T = 'C(chrom_type, Treatment(reference="control"))'
        formula += f" + {T}"
        if type_interactions:
            formula += f" + confidence:{T} + threshold:{T}"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm(formula, data=d, groups=d["participant_id"],
                            vc_formula={"chromatogram": "0 + C(chromatogram_id)"})
            res = m.fit(reml=True, method="lbfgs")
            return res, d, "participant + chromatogram random effects"
        except Exception:  # noqa: BLE001
            try:
                m = smf.mixedlm(formula, data=d, groups=d["participant_id"])
                res = m.fit(reml=True, method="lbfgs")
                return res, d, "participant random intercept only (chromatogram VC failed)"
            except Exception as exc:  # noqa: BLE001
                return None, d, f"model failed: {exc}"


def pretty_term(name):
    """Human-readable fixed-effect name (strips the C(chrom_type, ...) wrapper)."""
    import re
    m = re.search(r"\[T\.(\w+)\]", name)
    typ = m.group(1) if m else None
    if name.startswith("confidence:") and typ:
        return f"confidence × type={typ}"
    if name.startswith("threshold:") and typ:
        return f"threshold × type={typ}"
    if typ:
        return f"type={typ} (vs control)"
    return name


def extract_effects(res, want=("confidence", "threshold", "confidence:threshold", "trial_c")):
    """Return {effect: dict(coef, lo, hi, p)} for the requested fixed effects."""
    if res is None:
        return {}
    ci = res.conf_int()
    out = {}
    for name in want:
        if name in res.params.index:
            out[name] = dict(coef=float(res.params[name]),
                             lo=float(ci.loc[name, 0]), hi=float(ci.loc[name, 1]),
                             p=float(res.pvalues[name]))
    return out


# ── figures ─────────────────────────────────────────────────────────────────
def forest_plot(effects_by_outcome, outcome_labels, outdir, fname, title,
                effects=EFFECTS, alpha=0.05, p_adj_by_outcome=None):
    """
    One column per effect; rows are outcomes. Dot = coefficient, line = 95% CI.
    Filled/starred when significant (BH-FDR if p_adj supplied, else raw p).
    """
    os.makedirs(outdir, exist_ok=True)
    outs = list(effects_by_outcome.keys())
    n = len(outs)
    fig, axes = plt.subplots(1, len(effects), figsize=(4.6 * len(effects), 1.0 + 0.5 * n),
                             sharey=True)
    axes = np.atleast_1d(axes)
    ypos = np.arange(n)[::-1]
    for ax, eff in zip(axes, effects):
        for y, o in zip(ypos, outs):
            e = effects_by_outcome[o].get(eff)
            if not e:
                continue
            p_use = e["p"]
            if p_adj_by_outcome is not None:
                p_use = p_adj_by_outcome.get(o, {}).get(eff, e["p"])
            sig = not np.isnan(p_use) and p_use < alpha
            color = "#b30000" if sig else "#64748b"
            ax.plot([e["lo"], e["hi"]], [y, y], color=color, lw=2, zorder=2)
            ax.scatter([e["coef"]], [y], s=60 if sig else 40, color=color,
                       zorder=3, edgecolors="white", linewidths=0.8,
                       marker="D" if sig else "o")
            st = stars(p_use)
            if st:
                ax.text(e["hi"], y + 0.18, st, color=color, fontsize=10, ha="center")
        ax.axvline(0, color="#94a3b8", lw=1, linestyle="--", zorder=1)
        ax.set_title(EFFECT_LABEL.get(eff, eff), fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([outcome_labels.get(o, o).replace("\n", " ") for o in outs],
                            fontsize=9)
    fig.suptitle(title + "   (red ◆ = significant; line = 95% CI; 0 = no effect)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _cell_stats(df, outcome, unit_mean=False):
    """Observed mean+SE per 2x2 cell. If unit_mean, average within participant first."""
    rows = {}
    for cond, (cf, th) in FACTORS.items():
        sub = df[df["condition"] == cond]
        if unit_mean and "participant_id" in sub.columns:
            vals = sub.groupby("participant_id")[outcome].mean().dropna().values
        else:
            vals = sub[outcome].dropna().values
        if len(vals):
            rows[(cf, th)] = (np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals)),
                              len(vals))
    return rows


def interaction_plot(df, outcomes, outcome_labels, outdir, fname, title,
                     unit_mean=False):
    """2x2 interaction plot per outcome: x=threshold, two lines for confidence."""
    os.makedirs(outdir, exist_ok=True)
    ncols = min(4, len(outcomes))
    nrows = int(np.ceil(len(outcomes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, outcome in zip(axes, outcomes):
        cells = _cell_stats(df, outcome, unit_mean)
        for conf, color, lab in [(0, "#f19066", "confidence OFF"),
                                 (1, "#2e6da4", "confidence ON")]:
            xs, ys, es = [], [], []
            for th in (0, 1):
                if (conf, th) in cells:
                    m, se, _ = cells[(conf, th)]
                    xs.append(th)
                    ys.append(m)
                    es.append(1.96 * se)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", color=color, lw=2,
                            capsize=4, label=lab, markersize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["bars OFF", "bars ON"])
        ax.set_title(outcome_labels.get(outcome, outcome), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        ax.legend(fontsize=7)
    for ax in axes[len(outcomes):]:
        ax.axis("off")
    fig.suptitle(title + "   (parallel lines = no interaction; error bars = 95% CI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def trial_plot(df, outcome, res, outdir, fname, title):
    """Model-predicted outcome across trials for the four conditions."""
    if res is None or "trial_c" not in res.params.index:
        return None
    os.makedirs(outdir, exist_ok=True)
    tmin, tmax = int(df["trial"].min()), int(df["trial"].max())
    tc_mean = df["trial"].mean()
    trials = np.arange(tmin, tmax + 1)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for cond, (cf, th) in FACTORS.items():
        pred = pd.DataFrame({"confidence": cf, "threshold": th,
                             "trial_c": trials - tc_mean})
        try:
            y = res.predict(pred)
        except Exception:  # noqa: BLE001
            continue
        ax.plot(trials, y, color=COND_COLORS.get(cond, "#999"), lw=2.2, label=cond)
    ax.set_xlabel("trial (chromatogram number in sequence)")
    ax.set_ylabel(outcome)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def variance_components(res):
    """
    Decompose a fitted mixed model's variance into participant, chromatogram, and
    residual. Units are the outcome's squared units. Boundary estimates (a
    component pinned at 0) are reported as 0.0.
    """
    out = {"participant": 0.0, "chromatogram": 0.0, "residual": float(res.scale)}
    try:
        cov = np.asarray(res.cov_re)
        if cov.size:
            out["participant"] = float(cov.ravel()[0])
    except Exception:  # noqa: BLE001
        pass
    try:
        vc = getattr(res, "vcomp", None)
        if vc is not None and len(vc):
            out["chromatogram"] = float(np.sum(vc))
    except Exception:  # noqa: BLE001
        pass
    return out


def print_variance(res, outcome):
    """Show how much of the outcome's variance is participant vs chromatogram vs residual."""
    if res is None:
        return
    vc = variance_components(res)
    total = sum(vc.values())
    print(f"\n  Variance decomposition for {outcome} (how much each source contributes):")
    if total <= 0:
        print("    (variance components not estimable)")
        return
    for src in ("participant", "chromatogram", "residual"):
        v = vc[src]
        print(f"    {src:14} {v:.5f}   ({100*v/total:5.1f}% of total)")
    print(f"    -> chromatogram difficulty accounts for {100*vc['chromatogram']/total:.1f}% "
          f"of the variance; it is modelled as a crossed random effect, so the")
    print(f"       condition estimates above are already adjusted for which charts each "
          f"participant happened to see.")


def chrom_type(stem):
    for t in ("control", "drift", "noise", "tiny"):
        if t in str(stem):
            return t
    return "other"


TYPE_COLORS = {"control": "#4575b4", "drift": "#91bfdb",
               "noise": "#fdae61", "tiny": "#d73027", "other": "#999999"}


def difficulty_plot(df, outcome, outcome_label, outdir, fname, title):
    """Mean outcome per chromatogram (sorted) — 'which charts are hard', colored by type."""
    os.makedirs(outdir, exist_ok=True)
    g = df.groupby("chromatogram_id")[outcome].agg(["mean", "count", "std"]).dropna()
    if g.empty:
        return None
    g = g.sort_values("mean")
    colors = [TYPE_COLORS[chrom_type(s)] for s in g.index]
    ses = (g["std"] / np.sqrt(g["count"])).fillna(0).values
    fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(g)), 5.5))
    xs = np.arange(len(g))
    ax.bar(xs, g["mean"].values, yerr=1.96 * ses, color=colors, alpha=0.9,
           edgecolor="white", linewidth=0.5, capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(g.index, rotation=90, fontsize=7)
    ax.set_ylabel(f"mean {outcome_label} (across participants)")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLORS[t]) for t in
               ("control", "drift", "noise", "tiny")]
    ax.legend(handles, ["control", "drift", "noise", "tiny"], fontsize=8, title="type")
    ax.set_title(title + "  (lower = harder; error bars = 95% CI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def type_condition_plot(df, outcome, outcome_label, outdir, fname, title):
    """
    One panel per chromatogram type; within each, the 4-condition means (participant-
    averaged). Shows whether the condition ordering differs by difficulty.
    """
    os.makedirs(outdir, exist_ok=True)
    types = [t for t in ("control", "drift", "noise", "tiny")
             if t in set(df["chrom_type"])]
    if not types:
        return None
    fig, axes = plt.subplots(1, len(types), figsize=(3.4 * len(types), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    conds = [c for c in COND_ORDER if c in set(df["condition"])]
    for ax, typ in zip(axes, types):
        sub = df[df["chrom_type"] == typ]
        means, ses = [], []
        for c in conds:
            pm = sub[sub["condition"] == c].groupby("participant_id")[outcome].mean().dropna()
            means.append(pm.mean() if len(pm) else np.nan)
            ses.append(1.96 * pm.std(ddof=1) / np.sqrt(len(pm)) if len(pm) > 1 else 0)
        xs = np.arange(len(conds))
        ax.bar(xs, means, yerr=ses, color=[COND_COLORS.get(c, "#999") for c in conds],
               alpha=0.9, edgecolor="white", capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(conds, rotation=40, ha="right", fontsize=7)
        ax.set_title(typ, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"mean {outcome_label}")
    fig.suptitle(title + "   (compare the condition pattern across difficulty types)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_model(res, note, outcome, primary=False):
    print("\n" + ("=" * 92 if primary else "-" * 92))
    tag = "  [PRIMARY]" if primary else ""
    print(f"MODEL: {outcome}{tag}    (random effects: {note})")
    print("-" * 92)
    if res is None:
        print("  model did not fit.")
        return {}
    eff = extract_effects(res)
    for name in ("confidence", "threshold", "confidence:threshold", "trial_c"):
        if name in eff:
            e = eff[name]
            print(f"  {EFFECT_LABEL.get(name, name).splitlines()[0]:28} "
                  f"coef={e['coef']:+.4f}  95% CI[{e['lo']:+.4f}, {e['hi']:+.4f}]  "
                  f"p={e['p']:.4f} {stars(e['p'])}")
    # chromatogram type fixed effects (and type interactions), if present
    ci = res.conf_int()
    type_terms = [n for n in res.params.index
                  if "chrom_type" in n and n not in ("Group Var",)]
    main_types = [n for n in type_terms if not n.startswith(("confidence:", "threshold:"))]
    inter_types = [n for n in type_terms if n.startswith(("confidence:", "threshold:"))]
    if main_types:
        print("  -- chromatogram type (difficulty, vs control) --")
        for n in main_types:
            print(f"  {pretty_term(n):28} coef={res.params[n]:+.4f}  "
                  f"95% CI[{ci.loc[n,0]:+.4f}, {ci.loc[n,1]:+.4f}]  "
                  f"p={res.pvalues[n]:.4f} {stars(res.pvalues[n])}")
    if inter_types:
        print("  -- does the feature effect differ by difficulty? (type interactions) --")
        for n in inter_types:
            print(f"  {pretty_term(n):28} coef={res.params[n]:+.4f}  "
                  f"95% CI[{ci.loc[n,0]:+.4f}, {ci.loc[n,1]:+.4f}]  "
                  f"p={res.pvalues[n]:.4f} {stars(res.pvalues[n])}")
    return eff
