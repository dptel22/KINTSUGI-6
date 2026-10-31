"""Perf P1: streaming live source (cubesat/live_stream.py) is the batch run, tick for tick."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from conftest import TINY_SCENARIO

from cubesat.live_stream import LiveStream, resolve
from cubesat.scenario import Scenario, load_scenario
from cubesat.sim import run_scenario
from cubesat.sim_v2 import EngineV2

NOMINAL = Path("data/scenarios/nominal.json")


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
    if engine == "v2":
        replay = list(EngineV2(sc, stream.injections).run())
    else:
        replay = run_scenario(stream.scenario)
    assert replay[0] == stream.header and replay[-1] == stream.summary
    assert before[:149] == replay[1:150] and after == replay[151:-1]
    assert any(ln.get("event") for ln in after)


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
    stream = LiveStream(load_scenario(NOMINAL), lookahead=5).start()
    assert stream.wait_frame(4, 5.0)
    time.sleep(0.05)
    assert len(stream.frames) == 5  # nothing consumed: the producer waits
    stream.stop()
    assert stream.join(2.0)


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
