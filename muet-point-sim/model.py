"""Discrete-event model of the MUET Point bus system.

Entities
--------
* Passengers  : arrive at a terminal (non-homogeneous Poisson), wait in a FIFO queue,
                board the bus at the head of the terminal line.
* Buses       : cycle  terminal line -> (hold / wait for slot) -> boarding -> trip
                -> alighting + layover -> terminal line at the other end.
* Terminals   : 0 = A = STC (Hyderabad), 1 = B = Main Gate (Jamshoro).  Each has a
                single boarding bay, a passenger queue and a FIFO line of idle buses.

Random numbers use common random numbers (CRN): the arrival streams and the per-bus,
per-trip travel-time variates depend only on ``seed``, so different dispatch policies
are compared on identical demand and identical road conditions.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import Params, Period, Policy, profile_arrays
from des import Simulator

NMAX = 80   # max fleet size supported by the pre-drawn travel-time variates
KMAX = 80   # max trips per bus


# --------------------------------------------------------------------------- #
# Random inputs
# --------------------------------------------------------------------------- #
def make_streams(seed: int):
    ss = np.random.SeedSequence(seed)
    a, b, n, d = ss.spawn(4)
    return (np.random.default_rng(a), np.random.default_rng(b),
            np.random.default_rng(n), np.random.default_rng(d))


def gen_arrivals(profile, horizon: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Non-homogeneous Poisson arrivals on [0, horizon] by thinning (Lewis & Shedler)."""
    xs, ys = profile_arrays(profile)
    lam_max = float(ys.max()) * scale
    if lam_max <= 0.0:
        return np.empty(0)
    n = rng.poisson(lam_max * horizon)
    cand = np.sort(rng.uniform(0.0, horizon, n))
    u = rng.uniform(0.0, lam_max, n)
    return cand[u < scale * np.interp(cand, xs, ys)]


# --------------------------------------------------------------------------- #
# State containers
# --------------------------------------------------------------------------- #
class Terminal:
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.queue: deque = deque()   # arrival times of waiting passengers (FIFO)
        self.line: deque = deque()    # idle buses waiting for the bay (FIFO)
        self.front: Optional[int] = None   # bus occupying the bay (holding or boarding)
        self.front_ready = 0.0        # time the front bus reached the bay
        self.holding = False          # front bus is waiting for a dispatch decision
        self.token = 0                # invalidates stale timeout events
        self.last_start = -1e9        # last time boarding started (for static headway)
        self.q_area = 0.0             # integral of queue length over time
        self.q_last = 0.0
        self.max_q = 0
        self.denied_lead = 0          # leading queue entries already counted as "denied boarding"
        self.denied_total = 0
        self.bay_busy = 0.0
        self.arrived = 0
        self.boarded = 0


@dataclass
class RunResult:
    summary: Dict[str, float]
    waits: np.ndarray
    wait_term: np.ndarray
    wait_arr: np.ndarray
    qseries: np.ndarray        # columns: time, queue A, queue B  (1-minute samples)
    t_end: float
    q_area_total: float
    n_arrived: int
    n_boarded: int
    loads: np.ndarray
    events: int
    fleet_audit: int


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class BusSystem:
    def __init__(self, params: Params, policy: Policy, period: Period, seed: int) -> None:
        if params.fleet_size > NMAX:
            raise ValueError("fleet_size exceeds NMAX")
        self.p, self.pol, self.per = params, policy, period
        self.sim = Simulator()
        self.terms = [Terminal(0), Terminal(1)]
        ra, rb, rn, rd = make_streams(seed)
        # day-level random factors (mean 1): demand level and road conditions of this "day"
        s_d, s_t = math.sqrt(math.log(1 + params.day_demand_cv ** 2)), math.sqrt(math.log(1 + params.day_traffic_cv ** 2))
        self.day_demand = math.exp(s_d * rd.standard_normal() - s_d ** 2 / 2)
        self.day_traffic = math.exp(s_t * rd.standard_normal() - s_t ** 2 / 2)
        scale = params.demand_scale * self.day_demand
        self.arrivals = [
            gen_arrivals(period.rate_A, period.horizon, scale, ra),
            gen_arrivals(period.rate_B, period.horizon, scale, rb),
        ]
        self.z = rn.standard_normal((NMAX, KMAX))
        self.iu = rn.random((NMAX, KMAX))
        self.ie = rn.exponential(1.0, (NMAX, KMAX))
        self.trips_done = [0] * params.fleet_size
        self.ptr = [0, 0]
        # sigma of the lognormal trip time
        self._sig2 = math.log(1.0 + params.travel_cv ** 2)
        self._sig = math.sqrt(self._sig2)
        # records
        self.waits: List[float] = []
        self.wait_term: List[int] = []
        self.wait_arr: List[float] = []
        self.trip_log: List[tuple] = []     # (start_time, terminal, load)
        self.moves: List[tuple] = []        # (depart_time, travel_time)
        self.samples: List[tuple] = []
        self.idle_with_q = 0.0
        self.hold_total = 0.0
        self._stop_now = False

    # ---- helpers ----------------------------------------------------------- #
    def _headway(self, t: float) -> float:
        lo, hi = self.per.static_peak_window
        return self.pol.H_peak if lo <= t <= hi else self.pol.H_off

    def _touch(self, term: Terminal) -> None:
        now = self.sim.now
        term.q_area += len(term.queue) * (now - term.q_last)
        term.q_last = now

    def _travel_time(self, bus: int) -> float:
        k = min(self.trips_done[bus], KMAX - 1)
        self.trips_done[bus] += 1
        lo, hi = self.per.congestion_window
        m = self.p.mean_oneway_min * (self.p.congestion_factor if lo <= self.sim.now <= hi else 1.0)
        t = self.day_traffic * m * math.exp(self._sig * self.z[bus, k] - self._sig2 / 2.0)   # E[t] = m * day factor
        if self.iu[bus, k] < self.p.incident_prob:
            t += self.p.incident_mean_min * self.ie[bus, k]
        return t

    # ---- event handlers ---------------------------------------------------- #
    def _schedule_next_arrival(self, k: int) -> None:
        i = self.ptr[k]
        if i < len(self.arrivals[k]):
            self.sim.schedule_at(float(self.arrivals[k][i]), self._on_arrival, k)

    def _on_arrival(self, k: int) -> None:
        term = self.terms[k]
        self._touch(term)
        term.queue.append(self.sim.now)
        term.arrived += 1
        if len(term.queue) > term.max_q:
            term.max_q = len(term.queue)
        self.ptr[k] += 1
        self._schedule_next_arrival(k)
        if self.pol.kind == "dynamic":
            for t in self.terms:
                if t.holding:
                    self._decide(t)

    def _promote(self, term: Terminal) -> None:
        """Move the next idle bus from the line into the boarding bay."""
        if term.front is not None or not term.line:
            return
        term.front = term.line.popleft()
        term.front_ready = self.sim.now
        term.holding = True
        term.token += 1
        if self.pol.kind == "dynamic" and self.pol.max_hold > 0:
            self.sim.schedule(self.pol.max_hold, self._timeout, term, term.token)
        self._decide(term)

    def _timeout(self, term: Terminal, token: int) -> None:
        if term.front is not None and term.holding and token == term.token:
            self._decide(term)

    def _decide(self, term: Terminal) -> None:
        """Dispatch decision for the bus in the bay of ``term``."""
        if term.front is None or not term.holding:
            return
        now, pol, cap = self.sim.now, self.pol, self.p.capacity
        if pol.kind == "static":
            earliest = term.last_start + self._headway(term.last_start)
            if now >= earliest - 1e-9:
                self._go(term)
            else:
                self.sim.schedule(earliest - now, self._timeout, term, term.token)
            return
        # dynamic hold-and-go
        n = len(term.queue)
        other = self.terms[1 - term.idx]
        held = now - term.front_ready
        go = (
            n >= cap
            or n >= pol.min_load * cap
            or held >= pol.max_hold - 1e-9
            or (pol.rebal_thr is not None and len(other.queue) >= pol.rebal_thr)
        )
        if go:
            self._go(term)

    def _go(self, term: Terminal) -> None:
        """Start boarding: passengers present now (up to capacity) board the front bus."""
        now, cap = self.sim.now, self.p.capacity
        bus = term.front
        term.holding = False
        n_wait = len(term.queue)
        n = min(n_wait, cap)
        self._touch(term)
        for _ in range(n):
            a = term.queue.popleft()
            self.waits.append(now - a)
            self.wait_term.append(term.idx)
            self.wait_arr.append(a)
        term.boarded += n
        term.denied_lead = max(0, term.denied_lead - n)
        remaining = len(term.queue)
        term.denied_total += remaining - term.denied_lead   # left behind by a full bus
        term.denied_lead = remaining
        dwell = self.p.dwell_base_min + self.p.dwell_per_pax_min * n
        term.bay_busy += dwell
        term.last_start = now
        self.hold_total += now - term.front_ready
        if n_wait > 0:
            self.idle_with_q += now - term.front_ready
        self.trip_log.append((now, term.idx, n))
        self.sim.schedule(dwell, self._depart, term, bus)

    def _depart(self, term: Terminal, bus: int) -> None:
        tt = self._travel_time(bus)
        self.moves.append((self.sim.now, tt))
        dest = 1 - term.idx
        self.sim.schedule(tt, self._arrive, dest, bus)
        term.front = None
        self._promote(term)

    def _arrive(self, dest: int, bus: int) -> None:
        self.sim.schedule(self.p.alight_min + self.p.layover_min, self._ready, dest, bus)

    def _ready(self, dest: int, bus: int) -> None:
        term = self.terms[dest]
        term.line.append(bus)
        self._promote(term)

    def _sample(self) -> None:
        self.samples.append((self.sim.now, len(self.terms[0].queue), len(self.terms[1].queue)))
        if self.sim.now < self.per.horizon + self.p.max_drain_min:
            self.sim.schedule(1.0, self._sample)

    def _done(self) -> bool:
        return self.sim.now >= self.per.horizon and not self.terms[0].queue and not self.terms[1].queue

    # ---- run + summary ----------------------------------------------------- #
    def run(self) -> RunResult:
        N = self.p.fleet_size
        nA = self.pol.init_at_A if self.pol.init_at_A is not None else N // 2
        for b in range(N):
            self.terms[0 if b < nA else 1].line.append(b)
        for term in self.terms:
            self._promote(term)
        for k in (0, 1):
            self._schedule_next_arrival(k)
        self.sim.schedule(0.0, self._sample)
        self.sim.run(until=self.per.horizon + self.p.max_drain_min, stop=self._done)
        for term in self.terms:
            self._touch(term)
        return self._summarize()

    def _fleet_audit(self) -> int:
        """Number of buses currently accounted for (line + bay + in transit/alighting). Must equal N."""
        in_line = sum(len(t.line) for t in self.terms)
        in_bay = sum(1 for t in self.terms if t.front is not None)
        in_flight = 0
        for _, _, fn, args in self.sim._heap:
            if fn in (self._arrive, self._ready):
                in_flight += 1
        return in_line + in_bay + in_flight

    def _summarize(self) -> RunResult:
        p, per = self.p, self.per
        N, cap, heavy = p.fleet_size, p.capacity, self.per.heavy
        t_end = self.sim.now
        w = np.array(self.waits)
        wt = np.array(self.wait_term, dtype=int)
        wa = np.array(self.wait_arr)
        wh = w[wt == heavy] if w.size else w
        loads = np.array([x[2] for x in self.trip_log], dtype=float)
        starts = np.array([x[0] for x in self.trip_log])
        terms_of_trip = np.array([x[1] for x in self.trip_log], dtype=int)
        qs = np.array(self.samples, dtype=float)
        qh = qs[:, 1 + heavy]
        lo, hi = per.peak_window
        peak_mask = (qs[:, 0] >= lo) & (qs[:, 0] <= hi)
        # moving time clipped to t_end
        mv = np.array(self.moves) if self.moves else np.zeros((0, 2))
        moving = float(np.sum(np.clip(mv[:, 0] + mv[:, 1], None, t_end) - np.minimum(mv[:, 0], t_end))) if len(mv) else 0.0
        n_pax = int(w.size)

        def pct(a, q):
            return float(np.percentile(a, q)) if a.size else float("nan")

        h_trips = terms_of_trip == heavy
        l_trips = ~h_trips
        denied = sum(t.denied_total for t in self.terms)
        summary = {
            "n_pax": n_pax,
            "mean_wait": float(w.mean()) if n_pax else float("nan"),
            "p50_wait": pct(w, 50),
            "p95_wait": pct(w, 95),
            "max_wait": float(w.max()) if n_pax else float("nan"),
            "share_wait_gt10": float((w > 10).mean()) if n_pax else float("nan"),
            "share_wait_gt15": float((w > 15).mean()) if n_pax else float("nan"),
            "mean_wait_heavy": float(wh.mean()) if wh.size else float("nan"),
            "p95_wait_heavy": pct(wh, 95),
            "max_queue_heavy": float(self.terms[heavy].max_q),
            "mean_queue_heavy_peak": float(qh[peak_mask].mean()) if peak_mask.any() else float("nan"),
            "denied_share": denied / n_pax if n_pax else float("nan"),
            "trips_total": int(len(loads)),
            "load_factor_heavy": float(loads[h_trips].sum() / (cap * h_trips.sum())) if h_trips.any() else float("nan"),
            "load_factor_light": float(loads[l_trips].sum() / (cap * l_trips.sum())) if l_trips.any() else float("nan"),
            "moving_frac": moving / (N * t_end) if t_end > 0 else float("nan"),
            "idle_with_queue_bus_h": self.idle_with_q / 60.0,
            "bay_util_heavy": self.terms[heavy].bay_busy / t_end if t_end > 0 else float("nan"),
            "clear_time_after_horizon": max(0.0, t_end - per.horizon),
        }
        return RunResult(
            summary=summary, waits=w, wait_term=wt, wait_arr=wa, qseries=qs, t_end=t_end,
            q_area_total=sum(t.q_area for t in self.terms),
            n_arrived=sum(t.arrived for t in self.terms), n_boarded=sum(t.boarded for t in self.terms),
            loads=loads, events=self.sim.events_processed, fleet_audit=self._fleet_audit(),
        )


def run_once(params: Params, policy: Policy, period: Period, seed: int) -> RunResult:
    return BusSystem(params, policy, period, seed).run()
