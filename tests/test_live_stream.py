"""Perf P1: streaming live source (cubesat/live_stream.py) is the batch run, tick for tick."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from conftest import TINY_SCENARIO, patch_slow_engine

from cubesat.live_stream import LOOKAHEAD, LiveStream, resolve
from cubesat.scenario import Failure, FailureMode, Scenario, load_scenario
from cubesat.session import _inject_horizon
from cubesat.sim import RunCancelled, run_scenario
from cubesat.sim_v2 import EngineV2

NOMINAL = Path("data/scenarios/nominal.json")


@pytest.fixture(autouse=True)
def _stop_streams(monkeypatch):
    """Every stream a test starts is stopped afterwards (no idle producers leak into
    the thread-count checks of tests/test_session_cancel.py)."""
    started: list[LiveStream] = []
    real_start = LiveStream.start

    def start(self):
        started.append(self)
        return real_start(self)

    monkeypatch.setattr(LiveStream, "start", start)
    yield
    for stream in started:
        stream.stop()
    assert all(stream.join(2.0) for stream in started)


def _expected(mode: str, sat: int, t_now: float, dt: float, degrade_s: float) -> Failure:
    """The failure a click at t_now must mean, built independently of resolve(): a
    silent death at t_now, a self-report on the next tick, an EPS fade starting on the
    next tick (so its DEGRADE_START lands in a frame not yet shown)."""
    t = {"DEAD": t_now, "THRUSTER_OUT": t_now + dt, "EPS_DEGRADE": t_now + dt + degrade_s}
    return Failure(t=t[mode], sat_id=sat, mode=FailureMode(mode), degrade_s=degrade_s)


def _replay(sc: Scenario, failures: list[Failure]) -> list[dict]:
    if sc.engine == "v2":
        return list(EngineV2(sc, failures).run())
    return run_scenario(sc.model_copy(update={"failures": [*sc.failures, *failures]}))


def _drain(stream: LiveStream, start: int = 0, upto: int | None = None) -> list[dict]:
    """Play the stream like the driver: read frame i, then consume (backpressure; the
    stream forgets played frames). Returns the frames played."""
    played, i = [], start
    while upto is None or i < upto:
        assert stream.wait_frame(i, 30.0)
        if i >= len(stream.frames):
            assert stream.done and stream.error is None
            break
        played.append(stream.frames[i])
        i += 1
        stream.consume(i)
    return played


def _v2(sc: Scenario) -> Scenario:
    return sc.model_copy(update={"engine": "v2"})


@pytest.mark.parametrize("engine", ["v1", "v2"])
def test_stream_equals_batch_run(engine):
    sc = load_scenario(NOMINAL)
    sc = _v2(sc.model_copy(update={"duration_s": 400.0})) if engine == "v2" else sc
    stream = LiveStream(sc, lookahead=8).start()
    played = _drain(stream)
    assert [stream.header, *played, stream.summary] == run_scenario(sc)
    assert len(stream._ring) <= 8  # snapshots only for the buffered window
    stream.stop()
    assert stream.join(2.0)


@pytest.mark.parametrize("mode", ["DEAD", "THRUSTER_OUT", "EPS_DEGRADE"])
@pytest.mark.parametrize("engine", ["v1", "v2"])
def test_inject_equals_fresh_run_with_that_failure(engine, mode):
    """Inject while playback stands at t_now with frames buffered ahead: the stream,
    replayed from zero with its injections, is the same run; nothing before the
    failure's first tick changed."""
    sc = load_scenario(NOMINAL).model_copy(update={"duration_s": 1200.0})
    sc = _v2(sc) if engine == "v2" else sc
    stream = LiveStream(sc, lookahead=12).start()
    before = _drain(stream, 0, 150)  # last emitted frame = 149
    assert stream.wait_frame(150 + 11, 30.0)  # the lookahead is full
    t_now = stream.times[149]
    failure, k_first = resolve(stream.scenario, 2, mode, t_now, 60.0)
    assert k_first in (149, 150)
    stream.inject(failure, k_first, 20)
    after = _drain(stream, 150)
    # against a fresh run with the INDEPENDENTLY built failure, not the stream's own
    replay = _replay(sc, [_expected(mode, 2, t_now, sc.dt_s, 60.0)])
    assert replay[0] == stream.header and replay[-1] == stream.summary
    assert before[:149] == replay[1:150] and after == replay[151:-1]
    assert any(ln.get("event") for ln in after)
    # and the stream's own record replays it too (v1: scenario; v2: EngineV2 contract)
    own = _replay(sc, stream.injections) if engine == "v2" else run_scenario(stream.scenario)
    assert own[151:-1] == after


def test_inject_infeasible_changes_nothing():
    """The trial horizon raises before commit: frames, scenario and engine untouched."""
    from cubesat.allocate import InfeasibleFormation

    sc = Scenario.model_validate(
        dict(TINY_SCENARIO, failures=[], allocation={"phase_sweep_deg": [90.0],
                                                     "dv_budget_mps": 0.01})
    )
    stream = LiveStream(sc, lookahead=4).start()
    played = _drain(stream, 0, 2)
    assert stream.wait_frame(5, 5.0)
    frames, scenario = list(stream.frames), stream.scenario
    failure, k_first = resolve(sc, 0, "DEAD", stream.times[1], 0.0)
    with pytest.raises(InfeasibleFormation):
        stream.inject(failure, k_first, 10)
    assert stream.frames == frames and stream.scenario is scenario
    played += _drain(stream, 2)
    assert [stream.header, *played, stream.summary] == run_scenario(sc)


def test_first_frame_is_one_tick_not_the_run():
    """live_24sat v1: the batch run is ~9 s (P0); the first frame must not wait for it."""
    stream = LiveStream(load_scenario(Path("data/scenarios/live_24sat.json"))).start()
    t0 = time.perf_counter()
    assert stream.wait_frame(0, 5.0)
    assert time.perf_counter() - t0 < 1.0  # generous ceiling (target 0.3 s), shared machine
    stream.stop()
    assert stream.join(2.0)


def test_stop_stops_the_producer_and_backpressure_bounds_it():
    assert LOOKAHEAD == 16  # P0: 10-20 ticks absorb the 40-100 ms replan / GC spikes
    stream = LiveStream(load_scenario(NOMINAL)).start()  # the default lookahead
    assert stream.wait_frame(LOOKAHEAD - 1, 5.0)
    time.sleep(0.05)
    assert len(stream.frames) == LOOKAHEAD  # nothing consumed: the producer waits
    stream.consume(4)
    assert stream.wait_frame(LOOKAHEAD + 3, 5.0)
    time.sleep(0.05)
    assert len(stream.frames) == LOOKAHEAD + 4
    stream.stop()
    assert stream.join(2.0)


def test_inject_cancel_stops_the_trial_promptly(monkeypatch):
    """A cancelled inject stops at its next trial tick, not at the end of its horizon,
    and commits nothing."""
    patch_slow_engine(monkeypatch, delay=0.2, after_inject=True)
    stream = LiveStream(load_scenario(NOMINAL)).start()
    _drain(stream, 0, 10)
    assert stream.wait_frame(10 + LOOKAHEAD - 1, 5.0)  # buffer full: the producer idles
    frames = list(stream.frames)
    failure, k_first = resolve(stream.scenario, 2, "DEAD", stream.times[9], 0.0)
    cancel, raised = threading.Event(), []

    def run():
        try:
            stream.inject(failure, k_first, 50, cancel)  # 50 x 0.2 s if never cancelled
        except RunCancelled as exc:
            raised.append(exc)

    worker = threading.Thread(target=run)
    t0 = time.perf_counter()
    worker.start()
    time.sleep(0.3)
    cancel.set()
    worker.join(5.0)
    assert raised and time.perf_counter() - t0 < 1.5
    assert stream.frames == frames and stream.injections == []


@pytest.mark.parametrize("engine", ["v1", "v2"])
def test_long_eps_fade_inject_keeps_only_the_lookahead(engine):
    """Review regression: the horizon runs through the fade to the replan (for the
    409), but snapshots and frames are kept only for one lookahead; the stream resumes
    from there and still is the fresh run."""
    sc = load_scenario(NOMINAL).model_copy(update={"duration_s": 1600.0})
    sc = _v2(sc) if engine == "v2" else sc
    stream = LiveStream(sc).start()
    before = _drain(stream, 0, 50)
    t_now = stream.times[49]
    failure, k_first = resolve(stream.scenario, 3, "EPS_DEGRADE", t_now, 1000.0)
    horizon = _inject_horizon(stream.scenario, failure, k_first)
    assert horizon > 500
    stream.inject(failure, k_first, horizon)
    assert len(stream._ring) <= LOOKAHEAD + 2
    assert len(stream.frames) <= k_first + LOOKAHEAD
    after = _drain(stream, 50)
    replay = _replay(sc, [_expected("EPS_DEGRADE", 3, t_now, sc.dt_s, 1000.0)])
    assert before[:49] == replay[1:50] and after == replay[51:-1]


def test_reassign_on_degrade_replay_limit_is_pinned():
    """Documented limit (cubesat/live_stream.py): a fresh run knows an injected
    failure from t = 0, so reassign_on_degrade's failing_soon set at an EARLIER fade
    differs from the live run's. Pinned so a fix (or a silent change) is noticed; with
    reassign_on_degrade off the same experiment replays exactly."""
    results, leader = {}, None
    for rod in (True, False):
        sc = load_scenario(Path("data/scenarios/eps_leader_6sat.json")).model_copy(
            update={"duration_s": 3000.0, "reassign_on_degrade": rod}
        )
        stream = LiveStream(sc).start()
        before = _drain(stream, 0, 500)
        if leader is None:  # the sat the degrade trigger handed the leader seat to
            leader = next(
                s["id"] for s in before[499]["sats"] if s["role"] == "leader" and s["id"] != 0
            )
        stream.inject(*resolve(stream.scenario, leader, "THRUSTER_OUT", stream.times[499],
                               0.0)[:2], 40)
        after = _drain(stream, 500)
        replay = run_scenario(stream.scenario)
        results[rod] = before[:499] == replay[1:500] and after == replay[501:-1]
    assert results == {True: False, False: True}


@pytest.mark.parametrize("engine", ["v1", "v2"])
def test_two_injects_at_the_same_t_now_both_apply(engine):
    """Review regression: a self-reported failure acts on k_now + 1, so the snapshot
    at k_now must learn about it before a DEAD at the same t_now forks from it."""
    sc = load_scenario(NOMINAL).model_copy(update={"duration_s": 600.0})
    sc = _v2(sc) if engine == "v2" else sc
    stream = LiveStream(sc, lookahead=8).start()
    before = _drain(stream, 0, 50)
    t_now = stream.times[49]
    for sat, mode in ((1, "THRUSTER_OUT"), (2, "DEAD")):
        stream.inject(*resolve(stream.scenario, sat, mode, t_now, 0.0)[:2], 20)
    after = _drain(stream, 50)
    assert [(f.sat_id, f.mode.value) for f in stream.injections] == [
        (1, "THRUSTER_OUT"), (2, "DEAD")
    ]
    replay = (
        list(EngineV2(sc, stream.injections).run()) if engine == "v2"
        else run_scenario(stream.scenario)
    )
    assert before[:49] == replay[1:50] and after == replay[51:-1]
