"""Run every experiment of the study and write results/*.csv, results/config_used.json, results/raw.pkl.

Usage:
    python run_experiments.py            # full study (about 2-4 minutes)
    python run_experiments.py --quick    # tiny smoke run (seconds)

Pipeline
    1. Simulation-based tuning of the dynamic policy (grid search, training seeds, robust objective)
    2. Final scenario comparison S0..S4 (independent test seeds, common random numbers)
    3. Paired differences vs. the baseline
    4. Sensitivity / what-if analysis (fleet size, demand level, trip-time variability)
    5. Fluid-model cross-check
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy import stats

from config import (BASE_PARAMS, PERIODS, S0, S1, S2, S3, SCENARIO_KEYS, Policy)
from fluid import fluid_queue, fluid_queue_day_avg
from model import run_once

OUT = "results"
METRICS = ["n_pax", "mean_wait", "p50_wait", "p95_wait", "max_wait", "share_wait_gt10", "share_wait_gt15",
           "mean_wait_heavy", "p95_wait_heavy", "max_queue_heavy", "mean_queue_heavy_peak", "denied_share",
           "trips_total", "load_factor_heavy", "load_factor_light", "moving_frac", "idle_with_queue_bus_h",
           "bay_util_heavy", "clear_time_after_horizon"]
QLEN = 301   # minutes of queue time-series kept for plots


def ci_half(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n))


def run_reps(params, policy, period, seeds):
    return [run_once(params, policy, period, s) for s in seeds]


def objective(results_by_period) -> float:
    """Tuning objective: mean over periods of (mean wait + 0.5 * P95 wait)."""
    vals = []
    for rs in results_by_period:
        vals.append(np.mean([r.summary["mean_wait"] + 0.5 * r.summary["p95_wait"] for r in rs]))
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
def tune(params, seeds, scales=(1.0, 1.25)):
    """Grid search over the dynamic-policy parameters, robust over nominal and +25 % demand."""
    holds = [0.5, 1.0, 2.0, 3.0]
    loads = [0.6, 0.8, 1.0]
    rebals = [None, 25, 50, 100, 200]
    rows = []
    for hold, load, reb in itertools.product(holds, loads, rebals):
        pol = Policy("grid", "dynamic", max_hold=hold, min_load=load, rebal_thr=reb)
        js = []
        for sc in scales:
            prm = replace(params, demand_scale=sc)
            js.append(objective([run_reps(prm, pol, per, seeds) for per in PERIODS]))
        rows.append({"max_hold": hold, "min_load": load, "rebal_thr": -1 if reb is None else reb,
                     **{f"J_scale_{sc}": j for sc, j in zip(scales, js)}, "J_robust": float(np.mean(js))})
    df = pd.DataFrame(rows).sort_values("J_robust").reset_index(drop=True)
    return df


def policy_from_row(row) -> Policy:
    reb = None if int(row["rebal_thr"]) < 0 else int(row["rebal_thr"])
    return Policy("S4 Tuned dynamic", "dynamic", max_hold=float(row["max_hold"]),
                  min_load=float(row["min_load"]), rebal_thr=reb)


# --------------------------------------------------------------------------- #
def main(quick: bool = False) -> None:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    R_final = 5 if quick else 30
    R_train = 3 if quick else 10
    R_sens = 4 if quick else 20
    train_seeds = list(range(1, 1 + R_train))
    test_seeds = list(range(1000, 1000 + R_final))
    sens_seeds = list(range(2000, 2000 + R_sens))

    # 1. tuning ---------------------------------------------------------------
    print("[1/5] tuning dynamic policy ...", flush=True)
    grid = tune(BASE_PARAMS, train_seeds)
    grid.to_csv(f"{OUT}/grid_search.csv", index=False)
    S4 = policy_from_row(grid.iloc[0])
    print("      best:", grid.iloc[0].to_dict(), flush=True)
    policies = {"S0": S0, "S1": S1, "S2": S2, "S3": S3, "S4": S4}

    # 2. final comparison -----------------------------------------------------
    print("[2/5] final scenario comparison ...", flush=True)
    raw = {"seeds": test_seeds, "policies": {k: asdict(v) for k, v in policies.items()}, "data": {}}
    rows, per_seed = [], []
    for per in PERIODS:
        for key, pol in policies.items():
            rs = run_reps(BASE_PARAMS, pol, per, test_seeds)
            waits = [r.waits.astype(np.float32) for r in rs]
            raw["data"][(per.name, key)] = {
                "waits": waits,
                "wait_term": [r.wait_term.astype(np.int8) for r in rs],
                "wait_arr": [r.wait_arr.astype(np.float32) for r in rs],
                "qseries": np.stack([np.pad(r.qseries[:QLEN, 1:3], ((0, max(0, QLEN - len(r.qseries[:QLEN]))), (0, 0)))
                                     for r in rs]),
            }
            for s, r in zip(test_seeds, rs):
                per_seed.append({"period": per.name, "scenario": key, "seed": s, **r.summary})
            for m in METRICS:
                x = np.array([r.summary[m] for r in rs], dtype=float)
                rows.append({"period": per.name, "scenario": key, "label": pol.name, "metric": m,
                             "mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)), "ci95": ci_half(x)})
    pd.DataFrame(rows).to_csv(f"{OUT}/summary_by_scenario.csv", index=False)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(f"{OUT}/per_replication.csv", index=False)

    # 3. paired differences ---------------------------------------------------
    print("[3/5] paired differences ...", flush=True)
    prow = []
    for per in PERIODS:
        base = ps[(ps.period == per.name) & (ps.scenario == "S0")].sort_values("seed")
        for key in SCENARIO_KEYS[1:]:
            cur = ps[(ps.period == per.name) & (ps.scenario == key)].sort_values("seed")
            for m in ["mean_wait", "p95_wait", "max_wait", "max_queue_heavy", "share_wait_gt15", "denied_share",
                      "moving_frac", "load_factor_heavy"]:
                d = cur[m].values - base[m].values
                b = float(base[m].mean())
                prow.append({"period": per.name, "scenario": key, "metric": m, "baseline": b,
                             "scenario_mean": float(cur[m].mean()), "diff": float(d.mean()), "diff_ci95": ci_half(d),
                             "rel_change_pct": 100.0 * float(d.mean()) / b if b else float("nan")})
    pd.DataFrame(prow).to_csv(f"{OUT}/paired_vs_baseline.csv", index=False)

    # 4. sensitivity ------------------------------------------------------------
    print("[4/5] sensitivity analysis ...", flush=True)
    sens_pol = {"S0": S0, "S1": S1, "S3": S3, "S4": S4}

    def sweep(name, xs, make_params):
        out = []
        for x in xs:
            prm = make_params(x)
            for key, pol in sens_pol.items():
                vals = {"mean_wait": [], "p95_wait": [], "max_queue_heavy": [], "moving_frac": []}
                for per in PERIODS:
                    for r in run_reps(prm, pol, per, sens_seeds):
                        for m in vals:
                            vals[m].append(r.summary[m])
                out.append({"x": x, "scenario": key, **{f"{m}": float(np.mean(v)) for m, v in vals.items()},
                            "mean_wait_ci95": ci_half(np.array(vals["mean_wait"])),
                            "p95_wait_ci95": ci_half(np.array(vals["p95_wait"]))})
        pd.DataFrame(out).to_csv(f"{OUT}/sens_{name}.csv", index=False)
        return pd.DataFrame(out)

    fleets = [28, 32] if quick else [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
    s_fleet = sweep("fleet", fleets, lambda n: replace(BASE_PARAMS, fleet_size=int(n)))
    s_dem = sweep("demand", [0.8, 1.0, 1.2] if quick else [0.7, 0.85, 1.0, 1.15, 1.3],
                  lambda x: replace(BASE_PARAMS, demand_scale=float(x)))
    sweep("congestion", [1.0, 1.4] if quick else [1.0, 1.15, 1.3, 1.5, 1.7],
          lambda x: replace(BASE_PARAMS, congestion_factor=float(x)))

    # baseline-headway sensitivity: how brittle is a static timetable?
    hrows = []
    for H in ([2.0, 3.0] if quick else [1.75, 2.0, 2.25, 2.5, 2.75, 3.0]):
        pol = Policy(f"static H={H}", "static", H_peak=H, H_off=2 * H)
        for per in PERIODS:
            rs = run_reps(BASE_PARAMS, pol, per, sens_seeds)
            for m in ("mean_wait", "p95_wait", "max_queue_heavy"):
                hrows.append({"H_peak": H, "period": per.name, "metric": m,
                              "mean": float(np.mean([r.summary[m] for r in rs])),
                              "ci95": ci_half(np.array([r.summary[m] for r in rs]))})
    pd.DataFrame(hrows).to_csv(f"{OUT}/sens_headway.csv", index=False)

    # fleet-equivalence: smallest fleet at which each scenario matches the baseline mean wait at N = 40
    ref = float(s_fleet[(s_fleet.scenario == "S0") & (s_fleet.x == max(fleets) if quick else s_fleet.x == 40)]["mean_wait"].iloc[0])
    equiv = {}
    for key in sens_pol:
        d = s_fleet[s_fleet.scenario == key].sort_values("x")
        ok = d[d.mean_wait <= ref]
        equiv[key] = int(ok.x.min()) if len(ok) else None

    # 5. fluid cross-check ------------------------------------------------------
    print("[5/5] fluid-model cross-check ...", flush=True)
    frows = []
    for per in PERIODS:
        for key in ("S0", "S1", "S3", "S4"):
            _, _, f = fluid_queue(BASE_PARAMS, policies[key], per)
            fa = fluid_queue_day_avg(BASE_PARAMS, policies[key], per)
            sim_q = ps[(ps.period == per.name) & (ps.scenario == key)]["max_queue_heavy"]
            frows.append({"period": per.name, "scenario": key, "rho_peak_fluid": f["rho_peak"],
                          "mu_peak_per_min": f["mu_peak_per_min"], "lambda_peak_per_min": f["lambda_peak_per_min"],
                          "max_queue_fluid": f["max_queue"], "max_queue_fluid_dayavg": fa["max_queue"],
                          "max_queue_sim_mean": float(sim_q.mean()),
                          "max_queue_sim_ci95": ci_half(sim_q.values)})
    pd.DataFrame(frows).to_csv(f"{OUT}/fluid_vs_sim.csv", index=False)

    with open(f"{OUT}/config_used.json", "w") as fh:
        json.dump({"params": asdict(BASE_PARAMS), "periods": [asdict(p) for p in PERIODS],
                   "policies": {k: asdict(v) for k, v in policies.items()},
                   "replications": {"train": R_train, "final": R_final, "sensitivity": R_sens},
                   "fleet_equivalence_min_fleet_matching_baseline_wait": equiv,
                   "baseline_reference_mean_wait_min": ref, "quick": quick,
                   "runtime_s": round(time.time() - t0, 1)}, fh, indent=2, default=str)
    with open(f"{OUT}/raw.pkl", "wb") as fh:
        pickle.dump(raw, fh)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
