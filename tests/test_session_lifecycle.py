"""Session mutations are serialized (audit B-1/B-2/B-3).

Decisions: concurrent starts -> the LAST accepted start wins, earlier/queued
starts fail 409 "superseded"; a stale inject (its run was replaced during the
recompute) fails 409; two simultaneous injects both apply (the second waits and
recomputes on top); injection never rewinds playback (driver held during the
recompute)."""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import TINY_SCENARIO, make_session, patch_slow_engine, write_scenario

from cubesat.session import LiveSource, SessionError
from cubesat.sim import RunCancelled, run_scenario

_real_run = run_scenario


# ---- item 1a: concurrent starts ------------------------------------------


def test_three_concurrent_starts_leave_last_consistent(tmp_path, monkeypatch):
    write_scenario(tmp_path, "a.json", name="a")
    write_scenario(tmp_path, "b.json", name="b")
    write_scenario(
        tmp_path, "c.json", name="c", roles=["leader", "imager", "imager"],
        formation=dict(TINY_SCENARIO["formation"], n_slots=3),
    )
    delay = {"a": 0.3, "b": 0.2, "c": 0.1}  # the LAST start finishes FIRST

    def slow_run(scenario, cancel_event=None):
        if cancel_event.wait(delay[scenario.name]):
            raise RunCancelled("cancelled")
        return _real_run(scenario)

    monkeypatch.setattr("cubesat.session.run_scenario", slow_run)

    async def main():
        _, session = make_session(tmp_path)
        res = await asyncio.gather(
            session.start_live("a.json", None, 1.0, "even_split"),
            session.start_live("b.json", None, 1.0, "hungarian"),
            session.start_live("c.json", None, 1.0, "hungarian_keepout"),
            return_exceptions=True,
        )
        assert res[0].__class__ is SessionError and res[0].status_code == 409
        assert res[1].__class__ is SessionError and res[1].status_code == 409
        assert res[2] is None  # only the last accepted start succeeds
        assert session.mode == "live"
        assert session.enrich.scenario == "c"
        assert session.enrich.policy == "hungarian_keepout"
        assert session._source.scenario.name == "c"
        assert session._source.scenario.n_sats == 3
        assert session.hello()["policy"] == "hungarian_keepout"
        await session.reset()

    asyncio.run(main())



# ---- item 1b: stale inject -------------------------------------------------


def test_stale_inject_discarded_when_run_replaced(tmp_path, monkeypatch):
    write_scenario(tmp_path, "a.json", name="a")
    write_scenario(tmp_path, "b.json", name="b")
    # perf P1: the injection's trial horizon is the slow part (0.1 s per tick)
    patch_slow_engine(monkeypatch, delay=0.1, after_inject=True)

    async def main():
        _, session = make_session(tmp_path)
        await session.start_live("a.json", None, 1.0)
        await session.pause()
        inj = asyncio.create_task(session.inject_failure("SAT-1", "DEAD"))
        await asyncio.sleep(0.05)  # recompute in flight
        start_b = asyncio.create_task(session.start_live("b.json", None, 1.0))
        with pytest.raises(SessionError) as exc:
            await inj
        assert exc.value.status_code == 409
        await start_b
        assert session._source.scenario.name == "b"
        assert session.enrich.scenario == "b"
        assert list(session._source.scenario.failures) == []  # a's recompute not applied
        await session.reset()

    asyncio.run(main())



# ---- item 1c: simultaneous injects ---------------------------------------


def test_two_simultaneous_injects_both_apply(tmp_path):
    write_scenario(tmp_path, "a.json", name="a")

    async def main():
        _, session = make_session(tmp_path)
        await session.start_live("a.json", None, 1.0)
        await session.pause()
        await asyncio.gather(
            session.inject_failure("SAT-1", "DEAD"),
            session.inject_failure("SAT-2", "DEAD"),
        )
        assert {f.sat_id for f in session._source.scenario.failures} == {0, 1}
        await session.reset()

    asyncio.run(main())



# ---- item 1d: no rewind ---------------------------------------------------


def test_inject_never_rewinds_sim_time(tmp_path, monkeypatch):
    write_scenario(tmp_path, "long.json", name="long", duration_s=2000.0)
    real_wf = LiveSource.with_failure.__func__

    def slow_wf(cls, *args, **kw):
        time.sleep(0.3)  # the driver used to keep streaming meanwhile
        return real_wf(cls, *args, **kw)

    async def main():
        hub, session = make_session(tmp_path)
        q = hub.subscribe()
        times: list[float] = []

        async def collect():  # concurrent: the hub queue is bounded and drops states
            while True:
                m = await q.get()
                if m["type"] == "sim_state":
                    times.append(m["sim_time"])

        collector = asyncio.create_task(collect())
        await session.start_live("long.json", None, 1000.0)  # 0.3 s = 300 sim-s
        while session.sim_time is None or session.sim_time < 200.0:
            await asyncio.sleep(0.01)
        monkeypatch.setattr("cubesat.session.LiveSource.with_failure", classmethod(slow_wf))
        await session.inject_failure("SAT-2", "DEAD")
        assert session.sim_time < 1800.0, "inject must land mid-run"
        while not session.finished:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        collector.cancel()
        backwards = [(a, b) for a, b in zip(times, times[1:], strict=False) if b < a]
        assert not backwards, backwards[:3]
        assert times[-1] == 2000.0
        await session.reset()

    asyncio.run(main())
