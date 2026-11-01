"""Shared H3/H4 fixtures (BACKEND_SPEC v1.1 section 11).

The tiny scenario is a 2-sat, 20 s run that computes in ~15 ms yet exercises the
full demo arc: FAILURE_DETECTED at t=10, REPLANNING, ASSIGNMENT at t=14, SUMMARY.
"""

from __future__ import annotations

import json
from pathlib import Path

TINY_SCENARIO: dict = {
    "name": "tiny",
    "seed": 7,
    "duration_s": 20.0,
    "dt_s": 2.0,
    "chief": {"altitude_km": 500.0},
    "formation": {"n_slots": 2, "radial_amp_km": 1.0, "phase_offset_deg": 0.0},
    "a_max_mps2": 0.01,
    "gains": {"wn": 0.01, "zeta": 1.0},
    "sensor": {"fov_half_angle_deg": 1.5, "grid_step_km": 5.0, "grid_extent_km": 5.0},
    "roles": ["leader", "imager"],
    "failures": [{"t": 6.0, "sat_id": 0, "mode": "DEAD"}],
}

TINY_SCENARIO_NO_FAILURES: dict = dict(TINY_SCENARIO, failures=[])


def write_tiny_scenario(directory: Path, scenario: dict | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tiny.json"
    path.write_text(json.dumps(scenario or TINY_SCENARIO), encoding="utf-8", newline="\n")
    return path


def run_line(t: float, coverage: float) -> dict:
    """Minimal tick line for hand-built replay logs."""
    return {
        "v": 1,
        "t": t,
        "state": "NOMINAL",
        "coverage": coverage,
        "min_sep": 100.0 + t,
        "sats": [
            {"id": 0, "role": "leader", "status": "NOMINAL", "r": [1.0, 0.0, 0.0],
             "v": [0.0, 0.0, 0.0], "dv_used": 0.0},
            {"id": 1, "role": "imager", "status": "NOMINAL", "r": [0.0, 1.0, 0.0],
             "v": [0.0, 0.0, 0.0], "dv_used": 0.0},
        ],
    }


def tiny_log_lines() -> list[dict]:
    """INIT + 4 ticks + SUMMARY, in the layout cubesat.sim emits."""
    return [
        {"v": 1, "t": 0.0, "state": "INIT", "scenario": "tiny", "seed": 7, "n_sats": 2},
        run_line(0.0, 0.10),
        run_line(2.0, 0.20),
        run_line(4.0, 0.30),
        run_line(6.0, 0.40),
        {"v": 1, "t": 6.0, "state": "SUMMARY", "coverage_mean_last_orbit": 0.25},
    ]


def write_log(log_dir: Path, name: str, lines: list[dict]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
        encoding="utf-8",
        newline="\n",
    )
    return path


# ---- session-lifecycle test helpers (tests/test_session_*.py) -----------------


def make_session(tmp_path):
    from cubesat.session import Hub, Session

    hub = Hub()
    return hub, Session(
        scenario_dir=tmp_path / "scenarios", log_dir=tmp_path / "logs", hub=hub,
        min_emit_interval=0.0,
    )


def write_scenario(tmp_path, fname, **overrides):
    """TINY_SCENARIO_NO_FAILURES variant written as scenarios/<fname>."""
    d = tmp_path / "scenarios"
    d.mkdir(parents=True, exist_ok=True)
    body = dict(TINY_SCENARIO_NO_FAILURES, **overrides)
    (d / fname).write_text(json.dumps(body), encoding="utf-8", newline="\n")


def drain(q) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ---- streaming compute seam (perf P1, cubesat/live_stream.py) -------------------


class SlowEngine:
    """A real SimV1 whose step() sleeps `delay` s (always, or only once something was
    injected): stands in for an oversized run at the streaming seam, make_engine."""

    def __init__(
        self, scenario, delay: float = 0.05, after_inject: bool = False,
        slow_steps: float = float("inf"),
    ) -> None:
        from cubesat.sim import SimV1

        self.real = SimV1(scenario)
        self.scenario, self.grid = scenario, self.real.grid
        self.cov = self.real.cov
        self.delay, self.after_inject, self.injected = delay, after_inject, False
        self.slow_left = slow_steps  # slow steps left (counted from the injection)

    k = property(lambda self: self.real.k)
    n_ticks = property(lambda self: self.real.n_ticks)
    injections = property(lambda self: self.real.injections)

    def header(self) -> dict:
        return self.real.header()

    def step(self) -> dict:
        if (not self.after_inject or self.injected) and self.slow_left > 0:
            import time

            time.sleep(self.delay)
            self.slow_left -= 1
        return self.real.step()

    def inject(self, failure) -> None:
        self.injected = True
        self.real.inject(failure)

    def run(self, cancel_event=None):
        yield from self.real.run(cancel_event)


def patch_slow_engine(monkeypatch, **kwargs) -> list:
    """Every LiveStream built afterwards steps a SlowEngine; returns the streams' engines."""
    made: list = []

    def make(scenario):
        made.append(SlowEngine(scenario, **kwargs))
        return made[-1]

    monkeypatch.setattr("cubesat.live_stream.make_engine", make)
    producer_threads(baseline=True)  # producers other tests left idle do not count
    return made


_PRE_EXISTING: set = set()


def producer_threads(baseline: bool = False) -> list:
    """Live producer threads started since the last producer_threads(baseline=True)
    (idle producers of sessions other tests abandoned without a reset are excluded)."""
    import threading

    alive = [t for t in threading.enumerate() if t.name == "live-stream"]
    if baseline:
        _PRE_EXISTING.clear()
        _PRE_EXISTING.update(alive)
    return [t for t in alive if t not in _PRE_EXISTING]
