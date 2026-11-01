"""Perf P1 fix round: the session on top of the streaming source (cubesat/live_stream.py)."""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import TINY_SCENARIO, patch_slow_engine, write_scenario

from cubesat.live_stream import LiveStream
from cubesat.session import Hub, Session, SessionError

SPEED_MAX = 1e9


def _session(tmp_path, **kw) -> tuple[Hub, Session]:
    hub = Hub(maxsize=100_000)
    return hub, Session(tmp_path / "scenarios", tmp_path / "logs", hub, **kw)


def test_emit_cap_holds_at_max_speed_when_the_producer_is_the_bottleneck(
    tmp_path, monkeypatch
):
    """Review regression: the starved-producer poll re-anchored the pacing AND forgot
    the last emit, so every frame went out (396 sim_state/s at speed 1e9). At most the
    30 Hz cap now, and the burst still advances (F12 yields)."""
    write_scenario(tmp_path, "long.json", name="long", duration_s=4000.0)
    patch_slow_engine(monkeypatch, delay=0.002)  # ~500 ticks/s: the producer binds

    async def main():
        hub, session = _session(tmp_path)  # default min_emit_interval = 1/30 s
        q = hub.subscribe()
        await session.start_live("long.json", None, SPEED_MAX)
        t0, states, t_sim = time.perf_counter(), 0, []
        while time.perf_counter() - t0 < 1.5:
            try:
                msg = await asyncio.wait_for(q.get(), 0.5)
            except TimeoutError:
                continue
            if msg.get("type") == "sim_state" and msg.get("sats"):
                states += 1
                t_sim.append(msg["sim_time"])
        rate = states / (time.perf_counter() - t0)
        await session.reset()
        return rate, t_sim

    rate, t_sim = asyncio.run(main())
    assert rate <= 31.0, rate
    assert len(t_sim) >= 10 and t_sim[-1] - t_sim[0] > 500.0  # it is not frozen either


def test_inject_superseded_after_its_commit_is_409(tmp_path, monkeypatch):
    """The epoch check after the inject: a start/reset that lands while the injected
    trial commits must win (409), never the stale inject's 200."""
    write_scenario(tmp_path, "a.json", name="a")

    async def main():
        _, session = _session(tmp_path, min_emit_interval=0.0)
        await session.start_live("a.json", None, 1.0)
        await session.pause()
        real = LiveStream.inject

        def inject_then_superseded(self, *args, **kwargs):
            real(self, *args, **kwargs)
            session._epoch += 1  # a newer start/reset arrived as the commit landed

        monkeypatch.setattr(LiveStream, "inject", inject_then_superseded)
        with pytest.raises(SessionError) as exc:
            await session.inject_failure("SAT-1", "DEAD")
        assert exc.value.status_code == 409
        session.shutdown()

    asyncio.run(main())


def test_infeasible_scenario_failure_past_the_lookahead_stops_loudly(tmp_path):
    """A scenario failure whose replan is infeasible but lies beyond the first lookahead
    cannot be a 409 at start (streaming only computed 16 ticks): the stream ends with a
    critical event naming it, and every viewer is told the session stopped."""
    write_scenario(
        tmp_path, "late.json", name="late", duration_s=200.0,
        failures=[{"t": 100.0, "sat_id": 0, "mode": "DEAD"}],
        allocation={"phase_sweep_deg": [90.0], "dv_budget_mps": 0.01},  # see Phase A test
    )

    async def main():
        hub, session = _session(tmp_path, min_emit_interval=0.0)
        q = hub.subscribe()
        await session.start_live("late.json", None, SPEED_MAX)  # 200 OK: t = 100 not seen yet
        msgs = []
        while True:
            msg = await asyncio.wait_for(q.get(), 10.0)
            msgs.append(msg)
            if msg.get("type") == "hello" and msg.get("mode") == "stopped":
                break
        return session, msgs

    session, msgs = asyncio.run(main())
    stopped = [m for m in msgs if m.get("type") == "event" and m.get("level") == "critical"
               and m["message"].startswith("stream stopped")]
    assert len(stopped) == 1
    text = stopped[0]["message"]
    assert "InfeasibleFormation" in text and "(scenario cannot recover)" in text
    assert session.mode == "stopped" and session._source is None


def test_eps_degrade_longer_than_the_run_is_422(tmp_path):
    write_scenario(tmp_path, "a.json", name="a", duration_s=200.0)

    async def main():
        _, session = _session(tmp_path, min_emit_interval=0.0)
        await session.start_live("a.json", None, 1.0)
        await session.pause()
        with pytest.raises(SessionError) as exc:
            await session.inject_failure("SAT-1", "EPS_DEGRADE", degrade_s=201.0)
        assert exc.value.status_code == 422 and "run duration" in exc.value.detail
        await session.inject_failure("SAT-1", "DEAD", degrade_s=600.0)  # ignored for DEAD
        session.shutdown()

    asyncio.run(main())


def test_api_rejects_unbounded_degrade_s():
    from cubesat.wire import InjectRequest

    with pytest.raises(ValueError):
        InjectRequest(sat_id="SAT-1", mode="EPS_DEGRADE", degrade_s=1e9)
    assert TINY_SCENARIO["duration_s"] < 86400.0
