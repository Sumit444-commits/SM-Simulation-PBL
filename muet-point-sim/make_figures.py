"""Build all figures (results/figures/*.pdf|png) and LaTeX tables/macros (results/tables.tex, results/macros.tex)
from the CSV/pickle files written by run_experiments.py."""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PERIODS, rate_at

R, FIG = "results", "results/figures"
os.makedirs(FIG, exist_ok=True)
COL = {"S0": "#D55E00", "S1": "#E69F00", "S2": "#56B4E9", "S3": "#009E73", "S4": "#0072B2"}
LAB = {"S0": "S0 Static (baseline)", "S1": "S1 Static, denser", "S2": "S2 Dynamic dispatch",
       "S3": "S3 + rebalancing", "S4": "S4 Tuned dynamic"}
LS = {"S0": "-", "S1": "--", "S2": "-.", "S3": ":", "S4": "-"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "figure.dpi": 150})
summ = pd.read_csv(f"{R}/summary_by_scenario.csv")
paired = pd.read_csv(f"{R}/paired_vs_baseline.csv")
raw = pickle.load(open(f"{R}/raw.pkl", "rb"))
cfg = json.load(open(f"{R}/config_used.json"))
PN = {"morning": "Morning (07:00-10:00)", "afternoon": "Afternoon (13:30-16:30)"}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def clock(per, m):
    h, mi = map(int, per.clock0.split(":"))
    t = h * 60 + mi + m
    return f"{t // 60:02d}:{t % 60:02d}"


# 1 demand profile ----------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7, 2.2), sharey=True)
for ax, per in zip(axs, PERIODS):
    t = np.linspace(0, per.horizon, 200)
    ax.plot(t, [rate_at(per.rate_A, x) for x in t], color="#0072B2", label="STC (Hyderabad)")
    ax.plot(t, [rate_at(per.rate_B, x) for x in t], color="#D55E00", ls="--", label="Main Gate (Jamshoro)")
    ax.axvspan(*per.peak_window, color="grey", alpha=0.15, label="Peak hour")
    ax.set_xticks(range(0, 181, 30)); ax.set_xticklabels([clock(per, m) for m in range(0, 181, 30)])
    ax.set_title(PN[per.name]); ax.set_xlabel("Time of day")
axs[0].set_ylabel("Arrivals (pax/min)"); axs[0].legend(loc="upper left")
save(fig, "fig_demand")

# 2 queue time series -------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7, 2.4), sharey=True)
for ax, per in zip(axs, PERIODS):
    for k in ("S0", "S1", "S3", "S4"):
        q = raw["data"][(per.name, k)]["qseries"][:, :, per.heavy]
        m = q.mean(0)[:241]
        ax.plot(np.arange(len(m)), m, color=COL[k], ls=LS[k], label=LAB[k])
        if k == "S0":
            lo, hi = np.percentile(q, [10, 90], axis=0)
            ax.fill_between(np.arange(len(m)), lo[:241], hi[:241], color=COL[k], alpha=0.15)
    ax.axvspan(*per.peak_window, color="grey", alpha=0.12)
    ax.set_xticks(range(0, 241, 60)); ax.set_xticklabels([clock(per, m) for m in range(0, 241, 60)])
    ax.set_title(PN[per.name]); ax.set_xlabel("Time of day")
axs[0].set_ylabel("Queue at dominant terminal (pax)"); axs[0].legend(loc="upper left")
save(fig, "fig_queue")

# 3 wait CDF ----------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7, 2.4), sharey=True)
for ax, per in zip(axs, PERIODS):
    for k in ("S0", "S1", "S2", "S3", "S4"):
        w = np.sort(np.concatenate(raw["data"][(per.name, k)]["waits"]))
        ax.plot(w, np.arange(1, len(w) + 1) / len(w), color=COL[k], ls=LS[k], label=LAB[k])
    ax.axvline(15, color="grey", lw=0.8, ls=":")
    ax.set_xlim(0, 45); ax.set_title(PN[per.name]); ax.set_xlabel("Waiting time (min)")
axs[0].set_ylabel("Cumulative share of passengers"); axs[1].legend(loc="lower right")
save(fig, "fig_wait_cdf")

# 4 wait by arrival time ----------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7, 2.3), sharey=True)
bins = np.arange(0, 181, 10)
for ax, per in zip(axs, PERIODS):
    for k in ("S0", "S1", "S4"):
        d = raw["data"][(per.name, k)]
        w = np.concatenate(d["waits"]); a = np.concatenate(d["wait_arr"]); tm = np.concatenate(d["wait_term"])
        m = tm == per.heavy
        idx = np.digitize(a[m], bins) - 1
        y = [w[m][idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
        ax.plot(bins[:-1] + 5, y, color=COL[k], ls=LS[k], marker="o", ms=2.5, label=LAB[k])
    ax.axvspan(*per.peak_window, color="grey", alpha=0.12)
    ax.set_xticks(range(0, 181, 30)); ax.set_xticklabels([clock(per, m) for m in range(0, 181, 30)])
    ax.set_title(PN[per.name]); ax.set_xlabel("Passenger arrival time")
axs[0].set_ylabel("Mean wait (min)"); axs[0].legend(loc="upper left")
save(fig, "fig_wait_by_time")


# 5 metric bars with CI -----------------------------------------------------
def get(per, k, m, col="mean"):
    return float(summ[(summ.period == per) & (summ.scenario == k) & (summ.metric == m)][col].iloc[0])


fig, axs = plt.subplots(1, 3, figsize=(7, 2.3))
for ax, (m, ttl, yl) in zip(axs, [("mean_wait", "Mean wait", "min"), ("p95_wait", "95th-percentile wait", "min"),
                                  ("max_queue_heavy", "Peak queue", "passengers")]):
    x = np.arange(5); wd = 0.38
    for j, per in enumerate(("morning", "afternoon")):
        v = [get(per, k, m) for k in COL]; e = [get(per, k, m, "ci95") for k in COL]
        ax.bar(x + (j - 0.5) * wd, v, wd, yerr=e, color=[COL[k] for k in COL],
               alpha=1.0 if j == 0 else 0.55, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels(list(COL), fontsize=7); ax.set_title(ttl); ax.set_ylabel(yl)
axs[0].text(0.02, 0.95, "solid = morning, light = afternoon", transform=axs[0].transAxes, fontsize=6.5, va="top")
fig.tight_layout(); save(fig, "fig_metrics")

# 6 utilisation -------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7, 2.1))
for ax, (m, ttl) in zip(axs, [("moving_frac", "Fleet time in motion (share)"),
                              ("load_factor_heavy", "Load factor, dominant direction")]):
    x = np.arange(5); wd = 0.38
    for j, per in enumerate(("morning", "afternoon")):
        ax.bar(x + (j - 0.5) * wd, [get(per, k, m) for k in COL], wd, color=[COL[k] for k in COL],
               alpha=1.0 if j == 0 else 0.55, yerr=[get(per, k, m, "ci95") for k in COL], capsize=1.5,
               error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels(list(COL)); ax.set_title(ttl); ax.set_ylim(0, 1.05)
fig.tight_layout(); save(fig, "fig_util")

# 7 sensitivity -------------------------------------------------------------
sf, sd, sc = (pd.read_csv(f"{R}/sens_{n}.csv") for n in ("fleet", "demand", "congestion"))
sh = pd.read_csv(f"{R}/sens_headway.csv")
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
for ax, (d, xl) in zip(axs[:3], [(sf, "Fleet size (buses)"), (sd, "Demand multiplier"), (sc, "Peak trip-time multiplier")]):
    for k in ("S0", "S1", "S3", "S4"):
        e = d[d.scenario == k].sort_values("x")
        ax.plot(e.x, e.p95_wait, color=COL[k], ls=LS[k], marker="o", ms=2.5, label=k)
    ax.set_xlabel(xl)
axs[0].set_ylabel("95th-pct wait (min)"); axs[0].legend(ncol=2, fontsize=6.5)
axs[0].axvline(40, color="grey", lw=0.7, ls=":")
hh = sh[sh.metric == "p95_wait"].groupby("H_peak")["mean"].mean()
axs[3].plot(hh.index, hh.values, color=COL["S0"], marker="o", ms=2.5)
axs[3].axhline(float(np.mean([get(p, "S4", "p95_wait") for p in ("morning", "afternoon")])), color=COL["S4"], label="S4")
axs[3].set_xlabel("Static peak headway (min)"); axs[3].legend(fontsize=6.5)
fig.tight_layout(); save(fig, "fig_sensitivity")

# 8 grid-search heat-map ----------------------------------------------------
g = pd.read_csv(f"{R}/grid_search.csv")
best_load = g.iloc[0]["min_load"]
gg = g[g.min_load == best_load].pivot(index="rebal_thr", columns="max_hold", values="J_robust")
fig, ax = plt.subplots(figsize=(3.3, 2.2))
im = ax.imshow(gg.values, cmap="viridis_r", aspect="auto")
ax.set_xticks(range(len(gg.columns))); ax.set_xticklabels(gg.columns)
ax.set_yticks(range(len(gg.index))); ax.set_yticklabels(["off" if i < 0 else i for i in gg.index])
for i in range(gg.shape[0]):
    for j in range(gg.shape[1]):
        ax.text(j, i, f"{gg.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="w")
ax.set_xlabel("max hold (min)"); ax.set_ylabel("rebalancing threshold (pax)"); ax.grid(False)
ax.set_title(f"Robust objective J (min), min_load={best_load:g}", fontsize=7.5)
save(fig, "fig_grid")

# ----------------------------------------------------------------- LaTeX ---
def f1(x): return f"{x:.1f}"


def row(per, m, fmt=f1, ci=True):
    cells = []
    for k in COL:
        v, e = get(per, k, m), get(per, k, m, "ci95")
        cells.append(f"{fmt(v)}$\\pm${fmt(e)}" if ci else fmt(v))
    return " & ".join(cells)


T = []
for per in ("morning", "afternoon"):
    T.append(f"% ---- results table {per}")
    T.append("\\begin{tabular}{@{}lccccc@{}}\\toprule")
    T.append("Metric & S0 & S1 & S2 & S3 & S4 \\\\ \\midrule")
    for lab, m, fmt, ci in [("Mean wait (min)", "mean_wait", f1, True), ("P95 wait (min)", "p95_wait", f1, True),
                            ("Max wait (min)", "max_wait", f1, False),
                            ("Wait $>$15 min (\\%)", "share_wait_gt15", lambda x: f"{100 * x:.1f}", False),
                            ("Peak queue (pax)", "max_queue_heavy", lambda x: f"{x:.0f}", False),
                            ("Left behind (\\%)", "denied_share", lambda x: f"{100 * x:.0f}", False),
                            ("Fleet in motion (\\%)", "moving_frac", lambda x: f"{100 * x:.0f}", False),
                            ("Load factor, dom. dir. (\\%)", "load_factor_heavy", lambda x: f"{100 * x:.0f}", False)]:
        T.append(f"{lab} & {row(per, m, fmt, ci)} \\\\")
    T.append("\\bottomrule\\end{tabular}")
    open(f"{R}/table_{per}.tex", "w").write("\n".join(T[T.index(f'% ---- results table {per}') + 1:]) + "\n")
    T = []

fl = pd.read_csv(f"{R}/fluid_vs_sim.csv")
with open(f"{R}/table_fluid.tex", "w") as fh:
    fh.write("\\begin{tabular}{@{}llccccc@{}}\\toprule\nPeriod & Scen. & $\\rho_{pk}$ & Fluid $Q_{max}$ & Fluid (day-avg) & Sim. $Q_{max}$ \\\\ \\midrule\n")
    for _, r in fl.iterrows():
        fh.write(f"{r.period[:3].title()}. & {r.scenario} & {r.rho_peak_fluid:.2f} & {r.max_queue_fluid:.0f} & "
                 f"{r.max_queue_fluid_dayavg:.0f} & {r.max_queue_sim_mean:.0f}$\\pm${r.max_queue_sim_ci95:.0f} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")

with open(f"{R}/table_grid.tex", "w") as fh:
    fh.write("\\begin{tabular}{@{}cccc@{}}\\toprule\nHold (min) & Min load & Rebal. thr. & $J$ (min) \\\\ \\midrule\n")
    for _, r in g.head(5).iterrows():
        fh.write(f"{r.max_hold:g} & {r.min_load:g} & {'off' if r.rebal_thr < 0 else int(r.rebal_thr)} & {r.J_robust:.2f} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")


def pr(per, k, m):
    return paired[(paired.period == per) & (paired.scenario == k) & (paired.metric == m)].iloc[0]


mac = []
NM = {"S0": "SZero", "S1": "SOne", "S2": "STwo", "S3": "SThree", "S4": "SFour"}


def M(name, val):
    for a, b in NM.items():
        name = name.replace(a, b)
    mac.append(f"\\newcommand{{\\{name}}}{{{val}}}")


for per in ("morning", "afternoon"):
    p = per[:3]
    for k in COL:
        M(f"{k}{p}Mean", f1(get(per, k, "mean_wait")))
        M(f"{k}{p}Pnf", f1(get(per, k, "p95_wait")))
        M(f"{k}{p}Q", f"{get(per, k, 'max_queue_heavy'):.0f}")
        M(f"{k}{p}Mov", f"{100 * get(per, k, 'moving_frac'):.0f}")
        M(f"{k}{p}Lf", f"{100 * get(per, k, 'load_factor_heavy'):.0f}")
        M(f"{k}{p}Max", f1(get(per, k, "max_wait")))
        M(f"{k}{p}Gt", f"{100 * get(per, k, 'share_wait_gt15'):.0f}")
        M(f"{k}{p}Den", f"{100 * get(per, k, 'denied_share'):.0f}")
    for k in ("S1", "S2", "S3", "S4"):
        M(f"{k}{p}Red", f"{-pr(per, k, 'mean_wait').rel_change_pct:.0f}")
        M(f"{k}{p}Dci", f1(pr(per, k, "mean_wait").diff_ci95))
        M(f"{k}{p}Ddiff", f1(-pr(per, k, "mean_wait")["diff"]))
eq = cfg["fleet_equivalence_min_fleet_matching_baseline_wait"]
for k, v in eq.items():
    M(f"eq{k}", "n/a" if v is None else str(v))
M("eqRef", f1(cfg["baseline_reference_mean_wait_min"]))
b = cfg["policies"]["S4"]
M("fourHold", f"{b['max_hold']:g}"); M("fourLoad", f"{b['min_load']:g}")
M("fourReb", "off" if b["rebal_thr"] is None else str(int(b["rebal_thr"])))
M("repsFinal", str(cfg["replications"]["final"])); M("repsTrain", str(cfg["replications"]["train"]))
M("repsSens", str(cfg["replications"]["sensitivity"]))
d1 = sd[(sd.x == 1.15)].set_index("scenario")
for k in ("S0", "S1", "S3", "S4"):
    M(f"dem{k}Pnf", f1(d1.loc[k, "p95_wait"]))
d2 = sh[(sh.metric == "mean_wait")].groupby("H_peak")["mean"].mean()
M("hTwoMean", f1(d2[2.0])); M("hThreeMean", f1(d2[3.0]))
sfl = sf[(sf.x == 40) & (sf.scenario == "S0")].mean_wait.iloc[0]
M("fleetFlat", f1(sfl))
open(f"{R}/macros.tex", "w").write("\n".join(mac) + "\n")
print("figures + tables done:", sorted(os.listdir(FIG)))
