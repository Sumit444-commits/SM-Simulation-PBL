"""Deterministic fluid (rate-balance) approximation of the heavy-direction queue.

Used as an independent analytical cross-check of the simulation (verification) and
to explain *why* the static timetable bottlenecks:

    dQ/dt = lambda(t) - mu(t)    when Q > 0 or lambda > mu,   Q >= 0

with the service-rate (capacity) of the heavy terminal

    static  : mu(t) = min( C / H(t) ,  N*C / Tc(t) )
    dynamic : mu(t) = min( C / d_full , N*C / Tc(t) )

where Tc is the bus cycle time (two trips + alighting/layover + boarding + holding at
the light terminal), C the bus capacity, N the fleet size, H the headway and d_full the
boarding time of a full bus (bay limit).
"""
from __future__ import annotations

import numpy as np

from config import Params, Period, Policy, rate_at


def cycle_time(params: Params, t: float, period: Period, hold_light: float) -> float:
    lo, hi = period.congestion_window
    m = params.mean_oneway_min * (params.congestion_factor if lo <= t <= hi else 1.0)
    tau = m + params.incident_prob * params.incident_mean_min
    d_full = params.dwell_base_min + params.dwell_per_pax_min * params.capacity
    d_light = params.dwell_base_min + params.dwell_per_pax_min * 5
    return 2 * tau + 2 * (params.alight_min + params.layover_min) + d_full + d_light + hold_light


def service_rate(params: Params, policy: Policy, period: Period, t: float) -> float:
    """Passengers per minute the heavy terminal can carry at time t."""
    C, N = params.capacity, params.fleet_size
    if policy.kind == "static":
        lo, hi = period.static_peak_window
        H = policy.H_peak if lo <= t <= hi else policy.H_off
        # static timetable holds buses at *both* terminals -> average extra hold ~ H/2 at the light end
        supply = N * C / cycle_time(params, t, period, hold_light=H / 2.0)
        return min(C / H, supply)
    hold = 0.0 if policy.rebal_thr is not None else min(policy.max_hold, 2.0)
    supply = N * C / cycle_time(params, t, period, hold_light=hold)
    d_full = params.dwell_base_min + params.dwell_per_pax_min * C
    return min(C / d_full, supply)


def fluid_queue(params: Params, policy: Policy, period: Period, dt: float = 0.05):
    """Integrate the fluid queue at the heavy terminal.  Returns (t, Q, summary)."""
    prof = period.rate_A if period.heavy == 0 else period.rate_B
    T = period.horizon + params.max_drain_min
    t = np.arange(0.0, T, dt)
    Q = np.zeros_like(t)
    wait_in = 0.0
    for i in range(1, len(t)):
        lam = rate_at(prof, t[i - 1]) * params.demand_scale if t[i - 1] <= period.horizon else 0.0
        mu = service_rate(params, policy, period, t[i - 1])
        Q[i] = max(0.0, Q[i - 1] + (lam - mu) * dt)
        wait_in += Q[i] * dt
        if t[i] > period.horizon and Q[i] <= 0.0:
            t, Q = t[: i + 1], Q[: i + 1]
            break
    lam_peak = max(rate_at(prof, x) for x in np.linspace(*period.peak_window, 50)) * params.demand_scale
    mu_peak = service_rate(params, policy, period, sum(period.peak_window) / 2.0)
    return t, Q, {
        "max_queue": float(Q.max()),
        "rho_peak": lam_peak / mu_peak,
        "mu_peak_per_min": mu_peak,
        "lambda_peak_per_min": lam_peak,
        "clear_time": float(t[-1]),
    }


def fluid_queue_day_avg(params: Params, policy: Policy, period: Period, n: int = 41):
    """Fluid max-queue averaged over the day-to-day demand multiplier (lognormal, mean 1).

    Uses n equally spaced quantiles of the multiplier so the result is deterministic.
    """
    from dataclasses import replace
    from math import exp, log, sqrt
    from scipy.stats import norm

    s = sqrt(log(1 + params.day_demand_cv ** 2))
    qs, rhos = [], []
    for i in range(n):
        f = exp(s * norm.ppf((i + 0.5) / n) - s * s / 2)
        _, _, d = fluid_queue(replace(params, demand_scale=params.demand_scale * f), policy, period, dt=0.1)
        qs.append(d["max_queue"])
        rhos.append(d["rho_peak"])
    return {"max_queue": float(sum(qs) / n), "rho_peak": float(sum(rhos) / n)}
