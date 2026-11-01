"""Streaming live source (perf P1, audit 2026-11-01 F1/F2): one producer thread steps
the engine (sim.SimV1 or sim_v2.EngineV2) at most `lookahead` ticks ahead of playback,
so the first frame costs one tick instead of the whole run and an injection costs a
fixed horizon instead of a recompute from t = 0.

Same cursor surface as session.ReplaySource (header, frames, times, summary, skipped)
plus scenario, `done` / `error`. frames/times are mutated IN PLACE (the driver holds
the list objects); every mutation happens under `_cv`.

Snapshots: copy.deepcopy(engine) taken before each step, kept for the ticks still
inside the buffer (>= the last emitted frame). The read-only scenario and grid and the
append-only coverage list are shared, so a snapshot is O(n_sats), not O(k); `_fork`
gives the restored copy its own coverage prefix. Injection = fork the snapshot of the
tick where the failure first acts, inject, step a fixed horizon (so an infeasible
replan surfaces as an error BEFORE anything is committed), then swap it in and drop
the buffered frames after the last emitted one. `scenario` always carries every
resolved injection, so `run_scenario(stream.scenario)` replays the run from t = 0
(v1; for v2 EngineV2(original scenario, engine.injections) is the contract).
Known limit (pinned by tests/test_live_stream.py): a fresh run knows every failure
from t = 0, so the future-reading `failing_soon` sets of v1's oracle hook and of
`reassign_on_degrade` can differ from the live run, which learns of an injection only
when it happens; replay-from-zero is then not byte-exact (eps_leader_6sat).

Determinism: the engine never sees the wall clock; pacing stays in session.py.
"""

from __future__ import annotations

import copy
import threading
from functools import cached_property

from cubesat import geo_v2
from cubesat.scenario import Failure, FailureMode, Scenario
from cubesat.sim import RunCancelled, SimV1

# P0 (docs/audit/2026-11-01-perf.md 9): replan / GC spikes are 40-100 ms against
# 0.03-1.1 ms ticks, so 10-20 ticks of lookahead absorb them.
LOOKAHEAD = 16


def make_engine(scenario: Scenario):
    if scenario.engine == "v2":
        from cubesat.sim_v2 import EngineV2

        return EngineV2(scenario)
    return SimV1(scenario)


def _snapshot(engine):
    """Copy of `engine` sharing scenario, grid and the append-only coverage list."""
    cov = engine.cov
    memo = {id(engine.scenario): engine.scenario, id(engine.grid): engine.grid, id(cov): cov}
    return copy.deepcopy(engine, memo)


def _fork(snap):
    """Independent engine at snap.k: its own coverage list, cut to snap.k entries."""
    cov = snap.cov
    memo = {id(snap.scenario): snap.scenario, id(snap.grid): snap.grid, id(cov): cov[: snap.k]}
    return copy.deepcopy(snap, memo)


def resolve(scenario: Scenario, sat_id: int, mode: str, t_now: float, degrade_s: float):
    """(failure, k_first): the failure to inject for a click at t_now (the last emitted
    frame) and the tick where it first acts. Silent deaths act at t_now itself (frame t_now
    is already on screen; the next frames are those of a fresh run with the failure at
    t_now). Everything that announces itself -- a self-report, an EPS fade's
    DEGRADE_START -- acts on the NEXT tick so its event lands in a frame not yet shown,
    and an EPS fade starts there instead of in the streamed past (audit B-4)."""
    dt = scenario.dt_s
    k_now = round(t_now / dt)
    fmode = FailureMode(mode)
    if fmode == FailureMode.DEAD or (fmode == FailureMode.EPS_DEGRADE and degrade_s <= 0):
        return Failure(t=t_now, sat_id=sat_id, mode=fmode, degrade_s=degrade_s), k_now
    t_act = (k_now + 1) * dt
    t_fail = t_act + degrade_s if fmode == FailureMode.EPS_DEGRADE else t_act
    return Failure(t=t_fail, sat_id=sat_id, mode=fmode, degrade_s=degrade_s), k_now + 1


class LiveStream:
    def __init__(self, scenario: Scenario, lookahead: int = LOOKAHEAD) -> None:
        self.scenario = scenario
        self.lookahead = max(1, lookahead)
        self._engine = make_engine(scenario)
        self.header = self._engine.header()
        self.summary: dict | None = None
        self.frames: list[dict] = []
        self.times: list[float] = []
        self.skipped = 0
        self.done = False
        self.error: BaseException | None = None
        self._cv = threading.Condition()
        self._consumed = 0  # playback cursor (frames index), for backpressure
        self._ring: dict[int, object] = {}  # tick -> snapshot taken before stepping it
        self._stop = self._hold = self._busy = False
        self._inject_lock = threading.Lock()
        self._thread = threading.Thread(target=self._produce, name="live-stream", daemon=True)

    # -- session source surface ---------------------------------------------------
    @cached_property
    def earth_view(self) -> dict:
        return geo_v2.hello_fields(self.scenario) if self.scenario.engine == "v2" else {}

    @property
    def scenario_name(self) -> str | None:
        return self.scenario.name

    @property
    def seed(self) -> int | None:
        return self.scenario.seed

    @property
    def injections(self) -> list[Failure]:
        return list(self._engine.injections)

    # -- producer ------------------------------------------------------------------
    def start(self) -> LiveStream:
        self._thread.start()
        return self

    def stop(self) -> None:
        """Cancel: the producer exits before its next tick (no orphan CPU)."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    set = stop  # duck-types threading.Event for session._cancel_compute

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _produce(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(  # idle when done: an injection may revive the run
                    lambda: self._stop
                    or (
                        not self._hold
                        and not self.done
                        and len(self.frames) - self._consumed < self.lookahead
                    )
                )
                if self._stop:
                    return
                engine, self._busy = self._engine, True
            try:
                if engine.k < engine.n_ticks:
                    snap = _snapshot(engine)
                    line = engine.step()
                else:
                    snap, line = None, next(engine.run())  # k == n_ticks: SUMMARY only
            except BaseException as exc:  # noqa: BLE001 -- surfaced to the driver
                with self._cv:
                    self.error, self.done, self._busy = exc, True, False
                    self._cv.notify_all()
                continue
            with self._cv:
                self._busy = False
                if snap is None:
                    self.summary, self.done = line, True
                    self._cv.notify_all()
                    continue
                self._ring[engine.k - 1] = snap
                # times first: the driver reads len(frames), then times[i], unlocked
                self.times.append(float(line["t"]))
                self.frames.append(line)
                self._cv.notify_all()

    # -- consumer ------------------------------------------------------------------
    def wait_frame(self, i: int, timeout: float | None = None) -> bool:
        """Block until frame i exists (or the stream ended); False on timeout."""
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.frames) > i or self.done or self._stop, timeout
            )

    def consume(self, cursor: int) -> None:
        """Playback reached `cursor`: release backpressure, forget snapshots before the
        last emitted frame (an injection never restores further back than that).
        ponytail: played frames stay in `frames` (tests and the policy/predictor pins
        read the live history); dropping them is the audit-F6 GC fix, measured in
        GATES.md, and needs those readers moved to a replay first."""
        with self._cv:
            for k in range(max(self._consumed - 1, 0), cursor - 1):
                self._ring.pop(k, None)
            self._consumed = max(self._consumed, cursor)
            self._cv.notify_all()

    # -- injection -----------------------------------------------------------------
    def inject(
        self, failure: Failure, k_first: int, horizon_ticks: int,
        cancel: threading.Event | None = None,
    ) -> None:
        """Apply `failure` from tick k_first (>= the last emitted tick): frames >= k_first
        are recomputed, `horizon_ticks` of them before returning. Raises (and changes
        nothing) when the trial run raises, e.g. allocate.InfeasibleFormation, or when
        `cancel` is set or the stream stopped (RunCancelled, never a late commit)."""
        with self._inject_lock:  # one at a time, even one that outlived its caller
            self._inject(failure, k_first, horizon_ticks, cancel)

    def _inject(self, failure, k_first, horizon_ticks, cancel) -> None:
        with self._cv:
            self._hold = True
            self._cv.wait_for(lambda: not self._busy)
        try:
            if self.done and self.error is None and k_first >= len(self.frames):
                raise RuntimeError("run finished: cannot inject")
            base = self._ring.get(k_first)
            if base is None:
                if self._engine.k != k_first:
                    raise RuntimeError(f"no snapshot for tick {k_first}")
                base = self._engine
            trial = _fork(base)
            trial.inject(failure)
            # Snapshots and frames only for the first lookahead ticks; the rest of the
            # horizon is the 409 check alone, and the stream resumes from `keep`.
            snaps, lines, keep = {}, [], None
            while trial.k < min(trial.n_ticks, k_first + horizon_ticks):
                if self._stop or (cancel is not None and cancel.is_set()):
                    raise RunCancelled(f"injection cancelled before tick {trial.k}")
                if trial.k == k_first + self.lookahead:
                    keep = _fork(trial)
                if keep is None:
                    snaps[trial.k] = _snapshot(trial)
                    lines.append(trial.step())
                else:
                    trial.step()
            if keep is None:
                keep = trial
            with self._cv:  # commit
                if self._stop or (cancel is not None and cancel.is_set()):
                    raise RunCancelled("injection cancelled before its commit")
                resolved = keep.injections[-1]
                for k in sorted(self._ring):
                    if k >= k_first:
                        del self._ring[k]
                    else:  # a snapshot before the failure acts: same timeline + it
                        self._ring[k].inject(resolved)
                self._ring.update(snaps)
                del self.frames[k_first:], self.times[k_first:]
                self.frames.extend(lines)
                self.times.extend(float(line["t"]) for line in lines)
                self._engine, self.summary, self.done, self.error = keep, None, False, None
                failures = [*self.scenario.failures, resolved]
                self.scenario = self.scenario.model_copy(update={"failures": failures})
        finally:
            with self._cv:
                self._hold = False
                self._cv.notify_all()
