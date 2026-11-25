"""Configuration for the MUET Point bus-scheduling simulation.

IMPORTANT
---------
No measured MUET data was available when this project was built.  Every numeric
input below (demand profiles, trip times, fleet size, capacity ...) is an
*assumption* chosen to be plausible and to reproduce the symptoms described in
the problem statement (long peak-hour queues at the STC and Main Gate
terminals).  Replace them with field counts / GPS trip times before drawing
operational conclusions -- see ``load_profile_csv`` and the README.

Time unit everywhere: minutes, measured from the start of each simulated window
(morning window starts 07:00, afternoon window starts 13:30).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

Profile = Tuple[Tuple[float, float], ...]  # ((minute, passengers_per_minute), ...)


# --------------------------------------------------------------------------- #
# Demand-profile helpers
# --------------------------------------------------------------------------- #
def profile_arrays(profile: Profile):
    xs = np.array([p[0] for p in profile], dtype=float)
    ys = np.array([p[1] for p in profile], dtype=float)
    return xs, ys


def rate_at(profile: Profile, t: float) -> float:
    xs, ys = profile_arrays(profile)
    return float(np.interp(t, xs, ys))


def expected_count(profile: Profile, horizon: float, scale: float = 1.0) -> float:
    """Expected number of arrivals, i.e. the integral of lambda(t) over [0, horizon]."""
    xs, ys = profile_arrays(profile)
    t = np.linspace(0.0, horizon, 18001)
    y = np.interp(t, xs, ys)
    return scale * float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(t)))


def load_profile_csv(path: str, period: str, terminal: str) -> Profile:
    """Load an observed demand profile.

    CSV columns: ``period,terminal,minute,rate_per_min`` (terminal is ``A`` or ``B``).
    Example row: ``morning,A,60,24.0``.
    """
    pts = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["period"] == period and row["terminal"] == terminal:
                pts.append((float(row["minute"]), float(row["rate_per_min"])))
    if not pts:
        raise ValueError(f"no rows for period={period!r} terminal={terminal!r} in {path}")
    return tuple(sorted(pts))


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Period:
    """One simulated service window.  Terminal A = STC (Hyderabad), B = Main Gate (Jamshoro)."""

    name: str
    clock0: str                       # wall-clock label of minute 0
    rate_A: Profile                   # arrivals/min at terminal A
    rate_B: Profile                   # arrivals/min at terminal B
    heavy: int                        # index of the dominant-demand terminal (0=A, 1=B)
    peak_window: Tuple[float, float]  # demand peak (used for reporting)
    congestion_window: Tuple[float, float]    # road congestion multiplier applies when departing
    static_peak_window: Tuple[float, float]   # window where the static timetable uses H_peak
    horizon: float = 180.0            # passengers arrive during [0, horizon]


@dataclass(frozen=True)
class Params:
    """Physical system parameters (assumptions -- see module docstring)."""

    fleet_size: int = 40
    capacity: int = 50                # seated + standing passengers per Point bus
    mean_oneway_min: float = 25.0     # mean off-peak Hyderabad <-> Jamshoro trip
    travel_cv: float = 0.18           # coefficient of variation of trip time
    congestion_factor: float = 1.15   # trip-time multiplier inside congestion_window
    incident_prob: float = 0.04       # probability a trip suffers an incident delay
    incident_mean_min: float = 8.0    # mean (exponential) incident delay
    alight_min: float = 1.0           # unloading time at destination
    layover_min: float = 2.0          # driver turn-around before re-joining the terminal line
    dwell_base_min: float = 0.5       # fixed boarding overhead
    dwell_per_pax_min: float = 0.8 / 60.0  # 0.8 s per boarding passenger
    demand_scale: float = 1.0
    day_demand_cv: float = 0.12       # day-to-day variation of total demand (lognormal multiplier, mean 1)
    day_traffic_cv: float = 0.08      # day-to-day variation of road conditions (trip-time multiplier)
    max_drain_min: float = 240.0      # safety cap on post-horizon queue draining


@dataclass(frozen=True)
class Policy:
    """Dispatch policy.

    kind = "static"  : fixed minimum headway between departures (timetable).
    kind = "dynamic" : hold-and-go -- depart when full, when load >= min_load*capacity,
                       when held >= max_hold, or (rebalancing) when the opposite
                       terminal queue >= rebal_thr.
    """

    name: str
    kind: str
    H_peak: float = 2.0
    H_off: float = 4.0
    max_hold: float = 2.0
    min_load: float = 0.6
    rebal_thr: Optional[int] = None
    init_at_A: Optional[int] = None   # buses at terminal A at t=0 (default: half)


# --------------------------------------------------------------------------- #
# Default scenario set
# --------------------------------------------------------------------------- #
MORNING = Period(
    name="morning", clock0="07:00",
    rate_A=((0, 5), (30, 14), (60, 24), (90, 26), (120, 19), (150, 8), (180, 3)),
    rate_B=((0, 2.5), (180, 2.5)),
    heavy=0, peak_window=(60, 120),            # 08:00-09:00
    congestion_window=(45, 135), static_peak_window=(30, 150),
)
AFTERNOON = Period(
    name="afternoon", clock0="13:30",
    rate_A=((0, 2.5), (180, 2.5)),
    rate_B=((0, 4), (30, 20), (60, 25), (90, 25), (120, 18), (150, 7), (180, 3)),
    heavy=1, peak_window=(30, 120),            # 14:00-15:30
    congestion_window=(30, 135), static_peak_window=(30, 150),
)
PERIODS = (MORNING, AFTERNOON)

BASE_PARAMS = Params()

S0 = Policy("S0 Static timetable (baseline)", "static", H_peak=2.5, H_off=5.0)
S1 = Policy("S1 Static, denser timetable", "static", H_peak=2.0, H_off=4.0)
S2 = Policy("S2 Dynamic dispatch", "dynamic", max_hold=2.0, min_load=1.0, rebal_thr=None)
S3 = Policy("S3 Dynamic + rebalancing", "dynamic", max_hold=2.0, min_load=1.0, rebal_thr=50)
# S4 is produced by the grid search in run_experiments.py (parameters stored in results/config_used.json)
S4_DEFAULT = Policy("S4 Tuned dynamic", "dynamic", max_hold=1.0, min_load=1.0, rebal_thr=50)

SCENARIO_KEYS = ("S0", "S1", "S2", "S3", "S4")
