"""Session/driver tests (BACKEND_SPEC v1.1 sections 5.3, 9, 11).

cubesat/session.py has zero FastAPI imports, so these are plain asyncio tests.
Determinism on Windows (section 14.3): frame building is pure/sync
(wire.to_messages over the source's frames); the driver only adds pacing, and
tests neutralize it with speed=1e6 and min_emit_interval=0.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from conftest import (
    TINY_SCENARIO,
    TINY_SCENARIO_NO_FAILURES,
    patch_slow_engine,
    producer_threads,
    tiny_log_lines,
    write_log,
    write_tiny_scenario,
)

from cubesat.replay import load_jsonl
from cubesat.scenario import load_scenario
from cubesat.session import Hub, LiveSource, ReplaySource, Session, SessionError
from cubesat.sim import run_scenario
from cubesat.wire import SessionEnrich, to_messages

SPEED_MAX = 1e6  # pacing neutralized: anchor + t/1e6 is always in the past


def _messages(lines: list[dict], enrich: SessionEnrich) -> list[dict]:
    out: list[dict] = []
    for line in lines:
        out.extend(to_messages(line, enrich))
    return out


def _is_run_complete(msg: dict) -> bool:
    return msg.get("type") == "event" and msg.get("message") == "Run complete"


async def _drain_until_run_complete(q, session, timeout: float = 10.0) -> list[dict]:
    msgs: list[dict] = []
    while True:
        msg = await asyncio.wait_for(q.get(), timeout)
        msgs.append(msg)
        if _is_run_complete(msg):
            break
    assert session.finished, "driver must set finished at SUMMARY"
    return msgs


def _make_session(tmp_path, *, hub=None, min_emit_interval=0.0):
    hub = hub or Hub()
    session = Session(
        scenario_dir=tmp_path / "scenarios",
        log_dir=tmp_path / "logs",
        hub=hub,
        min_emit_interval=min_emit_interval,
    )
    return hub, session


def test_hub_slow_consumer_drops_state_keeps_events():
    """Section 9.4: overflow drops the oldest queued sim_state only; events and
    role_reassignment are never dropped, and publish never blocks."""

    async def main():
        hub = Hub(maxsize=4)
        q = hub.subscribe()
        state = {"type": "sim_state", "sim_time": 0.0}
        event = {"type": "event", "sim_time": 0.0, "level": "critical", "message": "x"}
        role = {"type": "role_reassignment", "sim_time": 0.0, "assignments": []}
        for i in range(4):  # fill the queue with sim_states
            hub.publish(dict(state, sim_time=float(i)))
        hub.publish(event)  # overflow: oldest sim_state drops, event survives
        kept = [q.get_nowait() for _ in range(q.qsize())]
        assert [m["sim_time"] for m in kept[:3]] == [1.0, 2.0, 3.0]  # order kept
        assert kept[-1]["type"] == "event"
        for i in range(4):  # re-fill: overflow again, role_reassignment arrives
            hub.publish(dict(state, sim_time=float(i) + 10.0))
        hub.publish(role)  # drops next oldest sim_state, keeps the role message
        kept = [q.get_nowait() for _ in range(q.qsize())]
        assert [m["type"] for m in kept] == [
            "sim_state", "sim_state", "sim_state", "role_reassignment",
        ]
        # degenerate flood: queue full of events, another event arrives --
        # publish must neither block nor drop a QUEUED event
        flood = Hub(maxsize=2)
        fq = flood.subscribe()
        flood.publish(event)
        flood.publish(dict(event, message="y"))
        flood.publish(dict(event, message="z"))  # must not raise or block
        assert fq.qsize() == 2
        assert {m["message"] for m in (fq.get_nowait(), fq.get_nowait())} == {"x", "y"}

    asyncio.run(main())


def test_session_replay_stream_and_seek(tmp_path):
    lines = tiny_log_lines()
    write_log(tmp_path / "logs", "tiny.jsonl", lines)
    expected = _messages(lines, SessionEnrich(scenario="tiny", seed=7))

    async def main():
        hub, session = _make_session(tmp_path)
        assert session.hello() == {
            "type": "hello", "mode": "stopped", "scenario": None, "seed": None,
            "sim_time": None, "skipped": 0,
        }
        q = hub.subscribe()
        await session.load_replay("tiny.jsonl", SPEED_MAX)
        assert session.mode == "replay"
        msgs = await _drain_until_run_complete(q, session)
        assert msgs == expected  # exact wire sequence, in order
        last_state = [m for m in msgs if m["type"] == "sim_state"][-1]
        assert session.snapshot() == last_state  # holds the last sim_state

        # backwards seek: first frame with t >= 1.0 is t=2.0 (no INIT resend)
        await session.seek(1.0)
        msgs = await _drain_until_run_complete(q, session)
        states = [m.get("sim_time") for m in msgs if m["type"] == "sim_state"]
        assert states == [2.0, 4.0, 6.0]
        # forwards seek: first frame with t >= 5.0 is t=6.0
        await session.seek(5.0)
        msgs = await _drain_until_run_complete(q, session)
        states = [m.get("sim_time") for m in msgs if m["type"] == "sim_state"]
        assert states == [6.0]
        await session.reset()
        assert session.mode == "stopped"
        assert session.snapshot()["sats"] == []  # empty-state from the INIT mapping

    asyncio.run(main())


def test_session_pause_resume_speed(tmp_path):
    lines = tiny_log_lines()
    write_log(tmp_path / "logs", "tiny.jsonl", lines)
    frame_times = [line["t"] for line in lines if line["state"] not in ("INIT", "SUMMARY")]

    async def main():
        hub, session = _make_session(tmp_path)
        q = hub.subscribe()
        msgs: list[dict] = []
        # real pacing: 2 s sim / 40x = 50 ms wall per frame
        await session.load_replay("tiny.jsonl", 40.0)
        await asyncio.sleep(0.13)  # a few frames have flowed
        await session.pause()
        while not q.empty():  # keep whatever crossed the cancel
            msgs.append(q.get_nowait())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.get(), 0.4)  # paused: nothing flows
        assert session.paused is True
        await session.resume()
        await session.set_speed(SPEED_MAX)
        while True:
            m = await asyncio.wait_for(q.get(), 10.0)
            msgs.append(m)
            if _is_run_complete(m):
                break
        assert session.finished, "driver must set finished at SUMMARY"
        # every frame exactly once, order preserved across the three re-anchors
        states = [m["sim_time"] for m in msgs if m["type"] == "sim_state"]
        assert states == [0.0] + frame_times  # leading 0.0 is the INIT mapping

    asyncio.run(main())


def test_live_equals_replay_frames(tmp_path):
    """Gate criterion 3 -- THE one-pipeline proof: streaming a scenario live and
    streaming its committed log produce identical message sequences."""
    scenario_path = write_tiny_scenario(tmp_path / "scenarios")
    lines = run_scenario(load_scenario(scenario_path))
    # the committed log: the same run, serialized exactly as cubesat.run does
    write_log(tmp_path / "logs", "tiny.jsonl", lines)

    async def main():
        live_hub, live = _make_session(tmp_path)
        live_q = live_hub.subscribe()
        await live.start_live("tiny.json", None, SPEED_MAX)
        live_msgs = await _drain_until_run_complete(live_q, live)

        replay_hub, replay = _make_session(tmp_path)
        replay_q = replay_hub.subscribe()
        await replay.load_replay("tiny.jsonl", SPEED_MAX)
        replay_msgs = await _drain_until_run_complete(replay_q, replay)

        assert live_msgs == replay_msgs
        # and the stream really covers the arc (failure + reassignment + summary)
        assert {m["type"] for m in live_msgs} == {
            "sim_state", "event", "role_reassignment", "run_summary"}

    asyncio.run(main())


def test_inject_prefix_byte_identical(tmp_path):
    """Gate criterion 4: injecting a failure at t_now recomputes a run whose
    frames before t_now are byte-identical to the pre-inject stream."""
    scenario = load_scenario(
        write_tiny_scenario(tmp_path / "scenarios", TINY_SCENARIO_NO_FAILURES)
    )
    before = run_scenario(scenario)  # the pre-inject stream (nominal)

    async def main():
        hub, session = _make_session(tmp_path)
        q = hub.subscribe()
        await session.start_live("tiny.json", None, 40.0)  # real pacing
        while session.sim_time is None or session.sim_time < 4.0:
            await asyncio.wait_for(q.get(), 10.0)
        assert session.sim_time == 4.0
        await session.inject_failure("SAT-2", "DEAD")  # t_now = 4.0
        assert session.mode == "live"

    asyncio.run(main())

    after = LiveSource.with_failure(scenario, "SAT-2", "DEAD", 4.0)
    lines_after = [after.header] + after.frames + [after.summary]
    prefix_before = [line for line in before if line.get("t", 0.0) < 4.0]
    prefix_after = [line for line in lines_after if line.get("t", 0.0) < 4.0]
    assert [json.dumps(line, sort_keys=True) for line in prefix_before] == [
        json.dumps(line, sort_keys=True) for line in prefix_after
    ]
    # and the failure actually changed the post-prefix run
    types_after = {(line.get("event") or {}).get("type") for line in lines_after}
    assert "FAILURE_DETECTED" in types_after


def test_session_compute_timeout_504_and_stopped(tmp_path, monkeypatch):
    """Section 7.1 compute-timeout row: a run that exceeds the wait_for window
    raises SessionError 504, tears the session back down to stopped, and leaves
    no driver task (the wall-clock timeout lives at the server, never in
    sim.py -- section 9.3). Perf P1: the window covers the first lookahead of
    the streaming producer, so the slow seam is the engine step."""

    async def main():
        hub, session = _make_session(tmp_path)
        write_tiny_scenario(tmp_path / "scenarios")
        session.run_timeout_s = 0.05
        patch_slow_engine(monkeypatch, delay=0.5)
        with pytest.raises(SessionError) as exc:
            await session.start_live("tiny.json", None, SPEED_MAX)
        assert exc.value.status_code == 504
        assert session.mode == "stopped"
        assert session._driver is None
        assert hub._queues == [] or all(q.empty() for q in hub._queues)

    asyncio.run(main())


def test_session_compute_timeout_cancels_compute_thread(tmp_path, monkeypatch):
    """Audit regression (medium): the run timeout used to abandon a still-
    running compute thread, so one oversized scenario could pin CPU in the
    shared server process after its own 504. The streaming producer is
    stopped on timeout and exits before its next tick."""

    async def main():
        hub, session = _make_session(tmp_path)
        write_tiny_scenario(tmp_path / "scenarios")
        session.run_timeout_s = 0.05
        patch_slow_engine(monkeypatch, delay=0.2)
        producer_threads(baseline=True)
        with pytest.raises(SessionError) as exc:
            await session.start_live("tiny.json", None, SPEED_MAX)
        assert exc.value.status_code == 504
        deadline = time.monotonic() + 2.0
        while producer_threads() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not producer_threads(), "orphan producer after the 504"

    asyncio.run(main())


def test_inject_timeout_cancels_recompute_thread(tmp_path, monkeypatch):
    """Audit regression (medium), inject path: an inject that outlives the
    window is a 504, its worker is cancelled before it commits (no late
    inject), and the old run plays on."""

    async def main():
        session = _make_session(tmp_path)[1]
        write_tiny_scenario(tmp_path / "scenarios", TINY_SCENARIO_NO_FAILURES)
        patch_slow_engine(monkeypatch, delay=0.2, after_inject=True)
        await session.start_live("tiny.json", None, speed=1.0)
        await session.pause()  # hold the cursor mid-run so inject is allowed
        session.run_timeout_s = 0.05
        source = session._source
        with pytest.raises(SessionError) as exc:
            await session.inject_failure("SAT-2", "DEAD")
        assert exc.value.status_code == 504
        await asyncio.sleep(0.5)  # the worker saw the cancel at its next tick
        assert source.scenario.failures == [] and source.injections == []
        session.shutdown()

    asyncio.run(main())


def test_run_scenario_cancel_event_unset_is_byte_identical(tmp_path):
    """The cancel hook is a pure read for every canonical caller: a run with a
    never-set event must be byte-identical to a run without one (AGENTS.md
    invariant 2 -- the golden pins depend on it)."""
    scenario = load_scenario(write_tiny_scenario(tmp_path / "scenarios"))
    without = [json.dumps(line, sort_keys=True) for line in run_scenario(scenario)]
    with_event = [
        json.dumps(line, sort_keys=True)
        for line in run_scenario(scenario, threading.Event())
    ]
    assert with_event == without


def test_run_scenario_cancelled_event_raises_no_partial_output(tmp_path):
    """A pre-set cancel event stops the run at the first tick boundary with
    RunCancelled -- never partial lines that could masquerade as a run."""
    from cubesat.sim import RunCancelled

    scenario = load_scenario(write_tiny_scenario(tmp_path / "scenarios"))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RunCancelled):
        run_scenario(scenario, cancel)


def test_session_driver_exception_never_silent(tmp_path, monkeypatch):
    """Section 7.1 belt-and-braces: if the driver task itself dies (any bug in
    frame building), the exception is surfaced as a critical event and the run
    is marked finished -- mode stays, but a dead driver never just stops the
    stream quietly."""
    lines = tiny_log_lines()
    write_log(tmp_path / "logs", "tiny.jsonl", lines)

    def boom(frame, enrich):
        raise KeyError("status")

    async def main():
        hub, session = _make_session(tmp_path)
        q = hub.subscribe()
        monkeypatch.setattr("cubesat.session.to_messages", boom)
        await session.load_replay("tiny.jsonl", SPEED_MAX)
        msg = await asyncio.wait_for(q.get(), 10.0)
        assert msg["type"] == "event"
        assert msg["level"] == "critical"
        assert "stream stopped" in msg["message"] and "KeyError" in msg["message"]
        assert session.mode == "replay"  # session stays addressable for reset/load
        assert session.finished, "dead driver must be terminal, not half-alive"

    asyncio.run(main())


def test_session_source_shapes_match(tmp_path):
    """LiveSource and ReplaySource expose the same frame surface -- the property
    that makes live-vs-replay drift structurally impossible (section 4)."""
    scenario = load_scenario(write_tiny_scenario(tmp_path / "scenarios", TINY_SCENARIO))
    live = LiveSource(scenario, run_scenario(scenario))
    replay = ReplaySource(
        load_jsonl(write_log(tmp_path / "logs", "t.jsonl", tiny_log_lines()))
    )
    for src in (live, replay):
        assert src.skipped == 0
        assert src.header["state"] == "INIT"
        assert src.summary["state"] == "SUMMARY"
        assert [f["t"] for f in src.frames] == src.times
    assert live.scenario_name == replay.scenario_name == "tiny"
    assert live.seed == replay.seed == 7
