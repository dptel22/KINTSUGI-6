"""`benchmark --mode perf`: where does a live run spend its time (audit 2026-11-01, phase P0).

Analysis layer only. Wall clock (time.perf_counter, the date stamp) is used HERE and nowhere in
the sim: the compute hook below is a cancel_event stand-in, so a run's log is byte-identical with
or without it (tests/v2/test_perf_bench.py). Per scenario: full compute, time to the first
sim_state with satellites through the REAL Session code path (in-process, no socket), inject
latency at 25/50/75 % of the run through the real Session.inject_failure, per-tick step cost,
to_messages cost and bytes per frame, optional cProfile top 15. One uvicorn + websockets run
measures what a real socket delivers per speed. Every timing is repeated; median and min/max kept.
"""

from __future__ import annotations

import asyncio
import cProfile
import hashlib
import json
import os
import platform
import pstats
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cubesat.live_stream import LiveStream
from cubesat.perf_ws import stats as _stats
from cubesat.perf_ws import websocket_section
from cubesat.scenario import Scenario, load_scenario
from cubesat.session import Hub, Session
from cubesat.sim import run_scenario
from cubesat.wire import SessionEnrich, to_messages

_pc = time.perf_counter
_ROOT = Path(__file__).resolve().parents[1]
NOTES = [
    "Wall-clock timings, shared machine (see machine_busy, load). Median of repeats; min/max kept.",
    "tick_ms_*, compute_s, setup_s: one instrumented run per repeat; the hook is a cancel_event "
    "stand-in stamping once per tick (setup_s = time before the first tick).",
    "first_frame_s: start_live() to the first sim_state with satellites, in-process Session, "
    "min_emit_interval=0 (so the 30 Hz cap cannot hide it; the socket run keeps the cap).",
    "inject_s: Session.inject_failure (DEAD, highest unscheduled sat id the replan "
    "accepts) to the first sim_state after it. The harness plays a fresh LiveStream to the stated "
    "percent off the clock (lookahead full), installs it as Session._source/_cursor/_last_frame_t "
    "(private) while the driver idles at speed 0.01; the inject is real.",
    "variant=true rows: committed v1 scenario with engine overridden in memory. Engine v2 has no "
    "recovery (no replan) yet, so its compute excludes the allocator sweep that dominates v1.",
    "cProfile inflates every number (deterministic profiler, one run): read the ranking only.",
    "json_us is ONE encode per frame; the server encodes every message again per connected client.",
]


class TickTimer:
    """Stands in for run_scenario's cancel_event: is_set() is called once per tick, never True."""

    def __init__(self) -> None:
        self.stamps: list[float] = []

    def is_set(self) -> bool:
        self.stamps.append(_pc())
        return False


class _TimedHub(Hub):
    def __init__(self) -> None:
        super().__init__()
        self.marks: list[float] = []

    def publish(self, msg: dict) -> None:
        if msg.get("type") == "sim_state" and msg.get("sats"):
            self.marks.append(_pc())
        super().publish(msg)

    def since(self, t0: float) -> float | None:
        return next((m - t0 for m in self.marks if m >= t0), None)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frames(lines: list[dict]) -> list[dict]:
    return [x for x in lines if x.get("state") not in ("INIT", "SUMMARY")]


def _wire_cost(lines: list[dict], scn: Scenario, repeats: int) -> dict:
    from cubesat.api import _wire_text

    frames, mean_us, p95_us, enc_us, sizes = _frames(lines), [], [], [], []
    for _ in range(repeats):
        en = SessionEnrich(scenario=scn.name, seed=scn.seed, policy="hungarian")
        tm, te, sz = [], [], []
        for f in frames:
            a = _pc()
            msgs = to_messages(f, en)
            b = _pc()
            texts = [_wire_text(m) for m in msgs]
            tm.append(b - a)
            te.append(_pc() - b)
            sz.append(sum(len(t.encode()) for t in texts))
        mean_us.append(np.mean(tm) * 1e6)
        p95_us.append(np.percentile(tm, 95) * 1e6)
        enc_us.append(np.mean(te) * 1e6)
        sizes = sz
    en = SessionEnrich(scenario=scn.name, seed=scn.seed, policy="hungarian")
    mid = len(frames) // 2
    if mid:
        to_messages(frames[mid - 1], en)
    peaks, tracemalloc_on = [], tracemalloc.is_tracing()
    tracemalloc.start()
    for f in frames[mid : mid + 50]:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        to_messages(f, en)
        peaks.append(tracemalloc.get_traced_memory()[1] - base)
    if not tracemalloc_on:
        tracemalloc.stop()
    return {
        "frames": len(frames),
        "to_messages_us": _stats(mean_us),
        "to_messages_us_p95": _stats(p95_us),
        "json_us": _stats(enc_us),
        "bytes_per_frame_mean": float(np.mean(sizes)),
        "bytes_per_frame_max": int(max(sizes)),
        "bytes_total": int(sum(sizes)),
        "alloc_peak_bytes_per_to_messages_mean": float(np.mean(peaks)) if peaks else None,
        "alloc_peak_sample_frames": len(peaks),
    }


def _profile(scn: Scenario, top: int = 15) -> list[dict]:
    pr = cProfile.Profile()
    pr.runcall(run_scenario, scn)
    st = pstats.Stats(pr)
    total = sum(v[2] for v in st.stats.values()) or 1.0

    def where(path: str) -> str:
        if path.startswith(str(_ROOT)):
            return os.path.relpath(path, _ROOT)
        if "site-packages/" in path:
            return path.split("site-packages/")[-1]
        return os.path.basename(path)

    rows = sorted(st.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:top]
    return [
        {"function": f"{where(k[0])}:{k[1]}({k[2]})", "ncalls": v[1], "tottime_s": round(v[2], 4),
         "cumtime_s": round(v[3], 4), "tottime_pct": round(100 * v[2] / total, 1)}
        for k, v in rows
    ]


async def _session_run(sdir: Path, fname: str, scn: Scenario) -> dict:
    hub = _TimedHub()
    s = Session(sdir, sdir / "logs", hub, min_emit_interval=0.0, run_timeout_s=3600.0)
    t0 = _pc()
    await s.start_live(fname, speed=0.01)  # driver emits frame 0, then idles ~200 s per frame
    out: dict = {"first_frame": hub.since(t0), "inject": {}, "errors": []}
    base = s._source.scenario  # policy seam etc. stay the session's own
    n_frames = int(round(base.duration_s / base.dt_s)) + 1
    dead = {f.sat_id for f in scn.failures}
    cands = [i for i in range(scn.n_sats - 1, -1, -1) if i not in dead][:3]  # highest ids first
    for pct in (25, 50, 75):
        idx = max(1, min(n_frames - 1, n_frames * pct // 100))
        out["inject"][pct] = None
        for sat in cands:  # a refused inject (see "errors") falls back to the next candidate
            s._cancel_driver()
            # streaming source played (consumed) up to frame idx - 1, as the driver would
            src = await asyncio.to_thread(_played_to, base, idx)
            if s._source is not src:
                s._source.stop()
            s._source, s._cursor, s._last_frame_t, s.finished = src, idx, src.times[idx - 1], False
            t0 = _pc()
            try:
                await s.inject_failure(f"SAT-{sat + 1}", "DEAD")
            except Exception as exc:  # noqa: BLE001 -- a failing inject is itself a finding
                out["errors"].append(f"{pct}% SAT-{sat + 1}: {type(exc).__name__}: {exc}")
                continue
            await asyncio.sleep(0)
            out["inject"][pct] = hub.since(t0)
            out.setdefault("sats", {})[str(pct)] = f"SAT-{sat + 1}"
            break
    await s.reset()
    return out


def _played_to(scenario: Scenario, idx: int) -> LiveStream:
    """A LiveStream whose playback consumed frames 0..idx-1 (lookahead buffered after)."""
    src = LiveStream(scenario).start()
    for i in range(idx):
        src.wait_frame(i)
        src.consume(i + 1)
    src.wait_frame(idx - 1 + src.lookahead)
    return src


def _measure(e: dict, repeats: int, sdir: Path, profile: bool) -> dict:
    scn, runs, lines = e["scn"], [], []
    for _ in range(repeats):
        tt, t0 = TickTimer(), _pc()
        lines = run_scenario(scn, tt)
        total = _pc() - t0
        d = np.diff(tt.stamps) * 1e3
        runs.append((total, tt.stamps[0] - t0, d.mean(), np.percentile(d, 95), d.max()))
    sess = [asyncio.run(_session_run(sdir, e["file"], scn)) for _ in range(repeats)]
    return {
        "id": e["id"], "name": scn.name, "engine": scn.engine, "variant": e["variant"],
        "variant_of": e["variant_of"], "v2_validates": e["v2_validates"], "n_sats": scn.n_sats,
        "n_ticks": len(_frames(lines)), "dt_s": scn.dt_s, "n_failures": len(scn.failures),
        "compute_s": _stats([r[0] for r in runs]), "setup_s": _stats([r[1] for r in runs]),
        "first_frame_s": _stats([x["first_frame"] for x in sess]),
        "inject_s": {str(p): _stats([x["inject"][p] for x in sess]) for p in (25, 50, 75)},
        "inject_sat": sess[0].get("sats", {}),
        "inject_errors": sorted({e for x in sess for e in x["errors"]}),
        "tick_ms_mean": _stats([r[2] for r in runs]), "tick_ms_p95": _stats([r[3] for r in runs]),
        "tick_ms_max": _stats([r[4] for r in runs]),
        "wire": _wire_cost(lines, scn, repeats),
        "profile": _profile(scn) if profile else None,
    }


# ---- report -------------------------------------------------------------------------------------
def _pytest_running() -> bool:
    cmd = ["ps", "-axo", "pid=,ppid=,command="]
    ps = subprocess.run(cmd, capture_output=True, text=True).stdout
    rows = [line.split(None, 2) for line in ps.splitlines() if line.strip()]
    parent = {int(r[0]): int(r[1]) for r in rows if len(r) == 3}
    mine, pid = set(), os.getpid()
    while pid in parent and pid not in mine:
        mine.add(pid)
        pid = parent[pid]
    return any("pytest" in r[2] and int(r[0]) not in mine for r in rows if len(r) == 3)


def _machine() -> dict:
    import scipy

    brand = ""
    if sys.platform == "darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip()
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": brand,
            "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__}


def _entries(sdir: Path, variants) -> tuple[list[dict], dict, list[str]]:
    entries, inputs, skipped = [], {}, []
    for path in sorted(sdir.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        try:
            scn = load_scenario(path)
        except ValueError as exc:
            skipped.append(f"{path.name}: {exc}".splitlines()[0])
            continue
        inputs[path.name] = sha256_file(path)
        try:
            v2 = Scenario.model_validate({**scn.model_dump(mode="json"), "engine": "v2"})
        except ValueError:
            v2 = None
        entries.append({"id": path.stem, "scn": scn, "file": path.name, "variant": False,
                        "variant_of": None, "v2_validates": v2 is not None or scn.engine == "v2"})
        if path.stem in variants and scn.engine != "v2" and v2 is not None:
            entries.append({"id": f"{path.stem}@v2", "scn": v2, "file": f"{path.stem}@v2.json",
                            "variant": True, "variant_of": path.name, "v2_validates": True})
    return entries, inputs, skipped


def loadavg() -> list[float] | None:
    """1/5/15-min load; None where the OS has none (Windows)."""
    return list(os.getloadavg()) if hasattr(os, "getloadavg") else None


def run_perf(sdir: Path, report: Path, *, repeats: int = 3, variants=("two_failure", "live_24sat"),
             profile=("two_failure", "two_failure@v2", "live_24sat"), ws_scenario="two_failure",
             ws_speeds=(1, 60, 1000, 1e6), ws_window_s=10.0, command: str = "") -> dict:
    sdir = Path(sdir)
    load0, busy_proc = loadavg(), _pytest_running()
    entries, inputs, skipped = _entries(sdir, variants)
    rows = []
    with tempfile.TemporaryDirectory(prefix="perf_scn_") as tmp:
        tdir = Path(tmp)
        for e in entries:
            (tdir / e["file"]).write_text(e["scn"].model_dump_json(), encoding="utf-8")
        for e in entries:
            t0 = _pc()
            rows.append(_measure(e, repeats, tdir, e["id"] in profile))
            print(f"perf: {e['id']:<24} {_pc() - t0:7.1f} s", file=sys.stderr, flush=True)
        ws = (websocket_section(sdir, ws_scenario, ws_speeds, ws_window_s, repeats)
              if (sdir / f"{ws_scenario}.json").is_file()
              else {"feasible": False, "reason": f"scenario {ws_scenario}.json not in {sdir}"})
    load1 = load0[0] if load0 else 0.0
    out = {
        "schema": "perf_baseline/1", "command": command, "date_utc": datetime.now(UTC).isoformat(),
        "repeats": repeats, "machine": _machine(),
        "load": {"start": load0, "end": loadavg()},
        "pytest_running_at_start": busy_proc, "machine_busy": bool(load1 > 2.0 or busy_proc),
        "inputs": {k: inputs[k] for k in sorted(inputs)}, "skipped_scenarios": skipped,
        "scenarios": rows, "websocket": ws, "notes": NOTES,
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


def run_cli(args, command: str) -> int:
    run_perf(args.base, args.report, repeats=args.repeats, command=command)
    print(f"wrote {args.report}")
    return 0
