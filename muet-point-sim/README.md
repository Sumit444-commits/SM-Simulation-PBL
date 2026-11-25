# MUET Point Bus System — Discrete-Event Simulation & Scheduling Optimization

PBL assignment: model, simulate and analyze the Mehran University (MUET) Point bus scheduling and queue
management system (STC and Main Gate terminals, Hyderabad ↔ Jamshoro), identify bottlenecks, and evaluate
algorithmic scheduling optimizations **without buying new buses**.

The full write-up (IEEE-style, with methodology, mathematical models, code snippets, graphs and
recommendations) is in [`report/main.pdf`](report/main.pdf).

> **Data caveat.** No measured MUET data was available. Every demand, trip-time and fleet number in
> `config.py` is an **assumption**, chosen to be plausible and to reproduce the symptoms in the problem
> statement. Results show *relative* behaviour of scheduling policies, not MUET measurements. To calibrate,
> replace the profiles in `config.py` (see "Using real data" below).

## What is modeled

* Two terminals: **A = STC (Hyderabad)**, **B = Main Gate (Jamshoro)**, one boarding bay and one FIFO passenger queue each.
* Passengers: non-homogeneous Poisson arrivals (peaks 08:00–09:00 and 14:00–15:30), day-to-day demand variation.
* Buses: closed loop of 40 buses (capacity 50), lognormal trip times with peak congestion, incidents, alighting/layover, boarding dwell.
* Policies (scenarios):

| ID | Policy |
|----|--------|
| S0 | Static timetable, 2.5 min peak / 5 min off-peak headway (baseline) |
| S1 | Static, denser timetable (2.0 / 4.0 min) |
| S2 | Dynamic hold-and-go: depart when full or after ≤ 2 min hold |
| S3 | S2 + rebalancing (depart immediately if the opposite queue ≥ 50) |
| S4 | Dynamic policy tuned by simulation-based grid search (train/test seeds separated) |

Metrics: mean / P95 / max wait, share waiting > 15 min, peak queue, share left behind by a full bus, fleet time in motion, load factor.
Analyses: 30 replications with common random numbers and paired 95 % CIs; sensitivity to fleet size, demand level, congestion and static headway; fluid-model cross-check.

## Repository layout

```
config.py           assumptions: demand profiles, fleet/travel parameters, scenario definitions
des.py              minimal discrete-event kernel (heap-based event calendar)
model.py            the bus-system model (terminals, buses, passengers, dispatch policies, metrics)
fluid.py            deterministic fluid-queue approximation (analytical cross-check)
run_experiments.py  tuning, scenario comparison, paired tests, sensitivity -> results/*.csv
make_figures.py     figures + LaTeX tables/macros from the results
test_model.py       verification tests (conservation, Little's law, analytic H/2 check, ...)
report/main.tex     IEEE-style report source (main.pdf = compiled report)
results/            CSV outputs, figures, config_used.json (raw.pkl is git-ignored)
```

## Run instructions

Requires Python 3.9+.

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

python -m unittest -v          # 9 verification tests (a few seconds)
python run_experiments.py      # full study, ~1 min  (use --quick for a 10 s smoke run)
python make_figures.py         # figures + LaTeX tables/macros into results/

cd report && pdflatex main.tex && pdflatex main.tex     # rebuild the PDF (needs a LaTeX install)
```

Everything is seeded: re-running gives identical numbers.

### Using real data

1. Count boardings per terminal in 5-minute bins and write a CSV with columns `period,terminal,minute,rate_per_min`
   (`period` = `morning` / `afternoon`, `terminal` = `A` / `B`, `minute` = minutes after 07:00 / 13:30).
2. In `config.py`, build the profiles with `load_profile_csv("demand.csv", "morning", "A")` and use them in `MORNING`/`AFTERNOON`.
3. Replace `Params` values (trip time, fleet size, capacity, dwell times) with measured ones, re-run the three commands above.

## Notes

* SimPy could not be installed in the build environment (no network), so a 30-line custom event kernel is used (`des.py`).
  The model logic is independent of the kernel and can be ported to SimPy.
* The report uses an IEEE-like two-column layout on the `article` class; swap in `IEEEtran.cls` if you have it.
* Fill in the author names on the title page of `report/main.tex`.
