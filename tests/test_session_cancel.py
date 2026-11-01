"""Abandoned compute threads are cancelled (audit B-5): reset, a superseded
start, a replaced inject, a client abort and shutdown all set the cancel Event
of the in-flight compute (since perf P1: the streaming producer)."""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import make_session, patch_slow_engine, producer_threads, write_scenario

from cubesat.session import SessionError

# ---- item 2: cancel abandoned work ----------------------------------------
# Perf P1: the compute seam is the streaming producer (cubesat/live_stream.py),
# so the blocker is a SlowEngine at make_engine instead of a patched run_scenario
# / LiveSource.with_failure; the pinned behaviour (no orphan thread, 409) is the same.


def _no_producer_within(seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while producer_threads():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


async def _until_producing():
    while not producer_threads():
        await asyncio.sleep(0.01)


def test_reset_cancels_in_flight_start(tmp_path, monkeypatch):
    write_scenario(tmp_path, "a.json", name="a")
    patch_slow_engine(monkeypatch, delay=0.2)  # a full lookahead takes ~3 s

    async def main():
        _, session = make_session(tmp_path)
        start = asyncio.create_task(session.start_live("a.json", None, 1.0))
        await _until_producing()
        t0 = time.monotonic()
        await session.reset()
        assert time.monotonic() - t0 < 2.0
        assert _no_producer_within(2.0), "producer still running after reset"
        with pytest.raises(SessionError) as exc:
            await start
        assert exc.value.status_code == 409
        assert session.mode == "stopped" and session._source is None

    asyncio.run(main())


def test_shutdown_and_abort_cancel_in_flight_start(tmp_path, monkeypatch):
    write_scenario(tmp_path, "a.json", name="a")
    patch_slow_engine(monkeypatch, delay=0.2)

    async def main():
        _, session = make_session(tmp_path)
        start = asyncio.create_task(session.start_live("a.json", None, 1.0))
        await _until_producing()
        session.shutdown()
        assert _no_producer_within(2.0)
        with pytest.raises(SessionError):
            await start
        # client abort (task cancelled) also stops the producer
        start = asyncio.create_task(session.start_live("a.json", None, 1.0))
        await _until_producing()
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start
        assert _no_producer_within(2.0)
        assert session.mode == "stopped"  # no zombie "live" session without a source

    asyncio.run(main())


def test_replaced_inject_is_cancelled(tmp_path, monkeypatch):
    write_scenario(tmp_path, "a.json", name="a")
    patch_slow_engine(monkeypatch, delay=0.2, after_inject=True)

    async def main():
        _, session = make_session(tmp_path)
        await session.start_live("a.json", None, 1.0)
        await session.pause()
        inj = asyncio.create_task(session.inject_failure("SAT-1", "DEAD"))
        await asyncio.sleep(0.1)  # the trial horizon is stepping, 0.2 s per tick
        assert not inj.done()
        t0 = time.monotonic()
        await session.reset()
        assert time.monotonic() - t0 < 2.0  # not parked behind the slow horizon
        assert _no_producer_within(2.0)
        with pytest.raises(SessionError) as exc:
            await inj
        assert exc.value.status_code == 409

    asyncio.run(main())


def test_app_shutdown_cancels_session(monkeypatch):
    """The API lifespan teardown calls Session.shutdown (SIGTERM must not wait
    for an abandoned compute)."""
    from fastapi.testclient import TestClient

    import cubesat.api
    from cubesat.session import Session

    calls = []
    monkeypatch.setattr(Session, "shutdown", lambda self: calls.append(self))
    cubesat.api._state = None
    with TestClient(cubesat.api.app):
        pass
    assert len(calls) == 1
    cubesat.api._state = None
