"""Verification tests for the MUET Point simulation.  Run:  python -m unittest -v"""
import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from config import (AFTERNOON, BASE_PARAMS, MORNING, Period, Policy, S0, S2, S3,
                    expected_count, load_profile_csv)
from fluid import fluid_queue
from model import run_once

QUIET = replace(BASE_PARAMS, day_demand_cv=0.0, day_traffic_cv=0.0)


class TestConservation(unittest.TestCase):
    def test_passengers_and_buses_conserved(self):
        for per in (MORNING, AFTERNOON):
            for pol in (S0, S2, S3):
                r = run_once(BASE_PARAMS, pol, per, seed=7)
                self.assertEqual(r.n_arrived, r.n_boarded)           # everybody is eventually served
                self.assertEqual(r.n_boarded, len(r.waits))
                self.assertEqual(int(r.loads.sum()), r.n_boarded)     # loads add up to boardings
                self.assertLessEqual(r.loads.max(), BASE_PARAMS.capacity)
                self.assertEqual(r.fleet_audit, BASE_PARAMS.fleet_size)  # no bus created or lost
                self.assertTrue((r.waits >= -1e-9).all())

    def test_littles_law_identity(self):
        """Integral of queue length == sum of individual waits  (L = lambda * W, exactly, per run)."""
        for pol in (S0, S3):
            r = run_once(BASE_PARAMS, pol, MORNING, seed=11)
            self.assertAlmostEqual(r.q_area_total, float(r.waits.sum()), delta=1e-6 * r.q_area_total)
            lam = r.n_boarded / r.t_end
            L = r.q_area_total / r.t_end
            self.assertAlmostEqual(L, lam * r.waits.mean(), delta=1e-6 * L)


class TestAnalytic(unittest.TestCase):
    def test_uniform_arrivals_fixed_headway(self):
        """Constant arrivals + ample buses + fixed headway H  =>  mean wait = H/2 (uniform arrival in a slot)."""
        flat = Period(name="flat", clock0="00:00", rate_A=((0, 6.0), (180, 6.0)), rate_B=((0, 6.0), (180, 6.0)),
                      heavy=0, peak_window=(0, 1), congestion_window=(0, 0), static_peak_window=(0, 0))
        prm = replace(QUIET, mean_oneway_min=5.0, travel_cv=1e-6, incident_prob=0.0,
                      alight_min=0.2, layover_min=0.3)
        H = 5.0
        pol = Policy("flat", "static", H_peak=H, H_off=H)
        means = [run_once(prm, pol, flat, seed=s).summary["mean_wait"] for s in range(5)]
        self.assertAlmostEqual(float(np.mean(means)), H / 2, delta=0.08)

    def test_fluid_model_orders_policies(self):
        _, _, base = fluid_queue(BASE_PARAMS, S0, MORNING)
        _, _, dyn = fluid_queue(BASE_PARAMS, S3, MORNING)
        self.assertGreater(base["max_queue"], dyn["max_queue"])
        self.assertGreater(base["rho_peak"], dyn["rho_peak"])

    def test_expected_arrivals_match_profile(self):
        counts = [run_once(QUIET, S0, MORNING, seed=s).n_arrived for s in range(20)]
        expected = expected_count(MORNING.rate_A, 180) + expected_count(MORNING.rate_B, 180)
        self.assertAlmostEqual(float(np.mean(counts)), expected, delta=0.03 * expected)


class TestPolicies(unittest.TestCase):
    def test_common_random_numbers(self):
        a = run_once(BASE_PARAMS, S0, MORNING, seed=5)
        b = run_once(BASE_PARAMS, S3, MORNING, seed=5)
        self.assertEqual(a.n_arrived, b.n_arrived)                    # identical demand across policies
        c = run_once(BASE_PARAMS, S0, MORNING, seed=5)
        self.assertTrue(np.array_equal(a.waits, c.waits))             # reproducible

    def test_dynamic_sends_full_buses_when_queue_is_long(self):
        r = run_once(replace(QUIET, demand_scale=1.3), S2, MORNING, seed=3)
        full_share = float((r.loads >= BASE_PARAMS.capacity).mean())
        self.assertGreater(full_share, 0.25)

    def test_static_respects_min_headway(self):
        r = run_once(BASE_PARAMS, S0, MORNING, seed=2)
        # fewer departures from the heavy terminal in a peak hour than 60/H_peak (+1 for edge effects)
        self.assertLessEqual(r.summary["trips_total"], 2 * (180 / 2.5) + 5)


class TestInputs(unittest.TestCase):
    def test_csv_profile_loader(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "demand.csv")
            with open(path, "w") as fh:
                fh.write("period,terminal,minute,rate_per_min\nmorning,A,60,24\nmorning,A,0,5\nmorning,B,0,2\n")
            self.assertEqual(load_profile_csv(path, "morning", "A"), ((0.0, 5.0), (60.0, 24.0)))
            with self.assertRaises(ValueError):
                load_profile_csv(path, "afternoon", "A")


if __name__ == "__main__":
    unittest.main()
