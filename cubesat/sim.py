"""Swarm simulation and versioned JSONL line model (gates G3/G4).

Pure core (MASTER_PROMPT 3.5): fixed-dt steps, no wall clock, no RNG, no
order-dependent iteration (AGENTS.md invariant 2); SI units, LVLH frame
(AGENTS.md invariants 3-4). cubesat.run is the I/O edge that writes these lines
as JSONL; two runs of one scenario + seed are byte-identical.

Log schema, one line per tick (MASTER_PROMPT section 5; v = LOG_SCHEMA_VERSION):

    {"v": 1, "t": <sim s>, "state": <str>,
     "sats": [{"id": int, "role": str, "status": str,
               "r": [x, y, z] m, "v": [vx, vy, vz] m/s, "dv_used": m/s}],
     "coverage": <fraction 0..1>, "min_sep": <m>,
     "event": {"type": <str>, "payload": <object>}}   # optional; omitted when None

The run opens with one metadata line (state INIT: v, t, scenario, seed, n_sats --
the G0 header, kept for replay) and closes with one SUMMARY line carrying the
time-averaged coverage over the last orbit (MASTER_PROMPT 3.4).

Swarm state machine (MASTER_PROMPT 3.5), logged per tick in `state`:
    NOMINAL -> FAILURE_DETECTED -> REPLANNING -> RECONFIGURING -> RECOVERED
A newly detected failure re-enters FAILURE_DETECTED from any state; with
recovery_enabled = False the machine stops at FAILURE_DETECTED (no-recovery
baseline). Events (type + payload, FIFO, at most one per tick line):
    FAILURE_DETECTED: {sat_id, mode, t_fail, t_detect, self_report?}
                      (self_report: true marks a failure the sat announced
                      itself -- a THRUSTER_OUT reports its own death on the
                      firing tick, so no heartbeat misses are needed and
                      t_detect == t_fail for tick-aligned failure times, as
                      in every committed scenario and live injection; a
                      mid-tick failure reports the enclosing tick)
    DEGRADE_START:    {sat_id, t_fail, degrade_s}  (EPS_DEGRADE fade onset)
    STATE:            {from, to}                    (plain transitions)
    ASSIGNMENT:       {assignments: [{sat_id, slot_index, role}], method,
                       cost_total, formation: {radial_amp_m, phase_offset_rad},
                       transfer_lower_bound_s}
    EVASIVE_MANEUVER: {sat_id, other_id, range_m, t}  (Addendum v1.2: a sat came
                       inside safety.evade_radius_m of another and started a
                       bounded away-from-intruder burn; default radius is below
                       every measured clearance, so canonical runs never emit it)
The replanner (cubesat.allocate.replan) runs BETWEEN sim steps -- one control
cycle per machine transition -- never inside the step loop below it. Every sat
emits a heartbeat each tick; a sat that stops reporting is declared DEAD after
missed_ticks_k misses. DEAD sats (and THRUSTER_OUT sats) coast ballistically --
never frozen -- and stay in the min_sep pairwise loop as collision hazards.

Failure modes (Addendum v1.2): EPS_DEGRADE with degrade_s > 0 puts the sat in
STATUS_DEGRADED from max(t_fail - degrade_s, 0) -- it keeps reporting, keeps
PD-holding its assigned seat, and still counts for coverage and the settle
check, but is excluded from replan assignable (status != NOMINAL). At t_fail it
becomes DEAD exactly like DEAD mode. degrade_s == 0 is sudden: the run matches
the same failure declared DEAD byte for byte except the FAILURE_DETECTED payload's
mode label, which honestly stays EPS_DEGRADE (one field, pinned by
test_eps_degrade_zero_window_equals_dead); a delayed DEGRADE_START event (the
FIFO pops at most one event per tick) is fine and documented.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections.abc import Iterator

import numpy as np

from cubesat.allocate import replan
from cubesat.coverage import coverage_fraction, footprint_radius, target_grid
from cubesat.formations import formation_slots
from cubesat.hcw import mean_motion, propagate
from cubesat.pd import PDController, bang_bang_time_lower_bound
from cubesat.scenario import Failure, FailureMode, Scenario

LOG_SCHEMA_VERSION = 1


class RunCancelled(RuntimeError):
    """Raised between ticks when run_scenario's optional cancel_event is set
    (server-side compute timeout only, cubesat/session.py). Never raised on any
    CLI/canonical path -- no partial output can ever be mistaken for a run."""
STATE_INIT = "INIT"
STATE_NOMINAL = "NOMINAL"
STATE_FAILURE_DETECTED = "FAILURE_DETECTED"
STATE_REPLANNING = "REPLANNING"
STATE_RECONFIGURING = "RECONFIGURING"
STATE_RECOVERED = "RECOVERED"
STATE_SUMMARY = "SUMMARY"
STATUS_NOMINAL = "NOMINAL"
STATUS_DEGRADED = "DEGRADED"
STATUS_DEAD = "DEAD"
STATUS_THRUSTER_OUT = "THRUSTER_OUT"

# Recovery tolerance: fraction of the largest assigned transfer distance.
_RECOVER_TOL_FRAC = 0.05

# Critical roles whose seat the proactive swap protects (oracle path and the
# degrade-trigger policy both target exactly these).
_CRITICAL_ROLES = ("leader", "downlink")


def _swap_critical_seat(
    holder: int,
    crit_role: str,
    failing_soon: set[int],
    assignable: list[int],
    roles: list[str],
    seat_of: list[int],
    t: float,
    mode_label: str | None,
) -> dict | None:
    """One proactive critical-seat swap: the failing holder exchanges seat AND
    role with the lowest-index assignable satellite that is non-critical and not
    failing soon (ties resolve to the lowest sat id, AGENTS.md 2). The oracle
    path's block, factored so the degrade-trigger policy can fire the SAME swap
    -- no duplicated logic. Returns the ORACLE_SWAP event to append, or None
    when no swap applies. mode_label (degrade trigger only) adds one additive
    payload key; oracle payloads keep their exact committed shape."""
    if holder not in failing_soon:
        return None
    candidates = [
        s
        for s in assignable
        if s not in failing_soon and roles[s] not in ("leader", "downlink")
    ]
    if not candidates:
        return None
    target = min(candidates)
    h_seat, t_seat = seat_of[holder], seat_of[target]
    seat_of[holder], seat_of[target] = t_seat, h_seat
    roles[holder], roles[target] = roles[target], roles[holder]
    payload: dict = {"role": crit_role, "from_sat": holder, "to_sat": target, "t": t}
    if mode_label is not None:
        payload["mode"] = mode_label
    return {"type": "ORACLE_SWAP", "payload": payload}


def build_slots(scenario: Scenario, n: float) -> list[np.ndarray]:
    """Slot initial states, sat i on slot i: phase = phase_offset_deg + 360*j/n_slots, on the
    `formation.shape` relative orbit (cubesat/formations.py). The default cartwheel has the
    cross-track amplitude equal to radial_amp_km (z in phase with x; the v1 slots, byte for
    byte), which spreads the ground (y, z) centers around a ring."""
    f = scenario.formation
    return formation_slots(
        f.shape, f.n_slots, f.radial_amp_km * 1000.0, n, math.radians(f.phase_offset_deg)
    )



class SimV1:
    """Engine v1 as a stepper (perf P1): the body of the former run_scenario loop,
    one tick per step(), state on the instance. Same surface as sim_v2.EngineV2:
    header(), step(), inject(), run(), k / n_ticks / injections; run_scenario is
    list(SimV1(scenario).run(cancel_event)), byte-identical to the old loop.
    No snapshot method: copy.deepcopy(engine) is the snapshot (cubesat/live_stream.py
    shares the read-only scenario/grid and the append-only coverage list)."""

    def __init__(self, scenario: Scenario) -> None:
        altitude_m = scenario.chief.altitude_km * 1000.0
        self.scenario = scenario
        self.n = n = mean_motion(altitude_m)
        self.dt = dt = scenario.dt_s
        self.n_sats = n_sats = scenario.n_sats
        slots = build_slots(scenario, n)
        self.states = [slot.copy() for slot in slots]
        self.controllers = [
            PDController(scenario.gains.wn, scenario.gains.zeta, scenario.a_max_mps2)
            for _ in scenario.roles
        ]
        self.r_f = footprint_radius(altitude_m, math.radians(scenario.sensor.fov_half_angle_deg))
        self.grid = target_grid(
            scenario.sensor.grid_step_km * 1000.0, scenario.sensor.grid_extent_km * 1000.0
        )
        # Mission bookkeeping. With no failures every default keeps this path
        # byte-identical to the G3 nominal log (golden-hash regression guard).
        self.formation = list(slots)  # current formation seat states (t = 0 convention)
        self.seat_of = list(range(n_sats))  # sat i -> seat index in `formation`
        self.roles = list(scenario.roles)
        self.status = [STATUS_NOMINAL] * n_sats
        self.missed = [0] * n_sats  # consecutive missed heartbeat ticks
        self.t_fail = [0.0] * n_sats
        self.fail_mode = [""] * n_sats
        self.declared = [False] * n_sats
        self.failures = sorted(scenario.failures, key=_failure_key)
        self.next_failure = 0
        # EPS_DEGRADE fade transitions, scheduled at max(t_fail - degrade_s, 0) --
        # the failure fires early as a DEGRADED status flip, in (fade time, sat_id)
        # order. Only degrade_s > 0 enters this schedule, which is what keeps
        # sudden failures (and every existing scenario) byte-identical.
        self.degrades = sorted(
            (
                f
                for f in scenario.failures
                if f.mode == FailureMode.EPS_DEGRADE and f.degrade_s > 0
            ),
            key=_degrade_key,
        )
        self.next_degrade = 0
        self.mission = STATE_NOMINAL
        self.events: list[dict] = []  # FIFO; at most one popped per tick line
        self.replan_due = False
        self.assigned = False
        self.recover_tol = 0.0
        self.oracle_warned = [False] * len(scenario.failures)
        # Evasion episodes (Addendum v1.2): per-sat countdown and held acceleration.
        # All-zero init keeps the default path byte-identical (see the evasion block).
        self.evade_ticks_left = [0] * n_sats
        self.u_evade = [np.zeros(3) for _ in range(n_sats)]
        self.t_first_detect: float | None = None
        self.n_ticks = int(round(scenario.duration_s / dt)) + 1
        self.coverage_by_tick: list[tuple[float, float]] = []
        self.k = 0
        self.injections: list[Failure] = []

    def header(self) -> dict:
        s = self.scenario
        return {
            "v": LOG_SCHEMA_VERSION,
            "t": 0.0,
            "state": STATE_INIT,
            "scenario": s.name,
            "seed": s.seed,
            "n_sats": s.n_sats,
        }

    def inject(self, failure: Failure) -> None:
        """Schedule `failure` (t >= the current tick) exactly where a fresh run with
        it in scenario.failures would hold it, so replaying the scenario with
        `injections` appended reproduces the live log. Not reproduced: the
        future-reading oracle / reassign_on_degrade `failing_soon` sets, which
        cannot know an injection before it happens."""
        if self.k >= self.n_ticks:
            raise RuntimeError("run finished: cannot inject")
        fades = failure.mode == FailureMode.EPS_DEGRADE and failure.degrade_s > 0
        onset = _degrade_key(failure)[0] if fades else failure.t
        if min(failure.t, onset) < self.k * self.dt - 1e-9:
            raise ValueError("injection (or its EPS fade onset) lies before the current tick")
        pos = bisect.bisect_right([_failure_key(f) for f in self.failures], _failure_key(failure))
        self.failures.insert(pos, failure)
        self.oracle_warned.insert(pos, False)
        if fades:
            keys = [_degrade_key(f) for f in self.degrades]
            self.degrades.insert(bisect.bisect_right(keys, _degrade_key(failure)), failure)
        failures = [*self.scenario.failures, failure]
        self.scenario = self.scenario.model_copy(update={"failures": failures})
        self.injections.append(failure)

    def step(self) -> dict:
        """Process tick k (fades, failures, heartbeats, log line), then the
        between-step orchestration and the physics step (skipped on the last tick)."""
        if self.k >= self.n_ticks:
            raise RuntimeError("run finished")
        k = self.k
        (
            scenario,
            n,
            dt,
            n_sats,
            states,
            controllers,
            r_f,
            grid,
            seat_of,
            roles,
            status,
            missed,
            t_fail,
            fail_mode,
            declared,
            failures,
            degrades,
            events,
            oracle_warned,
            evade_ticks_left,
            u_evade,
            n_ticks,
            coverage_by_tick,
        ) = (
            self.scenario,
            self.n,
            self.dt,
            self.n_sats,
            self.states,
            self.controllers,
            self.r_f,
            self.grid,
            self.seat_of,
            self.roles,
            self.status,
            self.missed,
            self.t_fail,
            self.fail_mode,
            self.declared,
            self.failures,
            self.degrades,
            self.events,
            self.oracle_warned,
            self.evade_ticks_left,
            self.u_evade,
            self.n_ticks,
            self.coverage_by_tick,
        )
        t = k * dt
        # EPS_DEGRADE fades fire (before this tick's heartbeat), in scheduled
        # order: the sat turns DEGRADED (still reporting, still seat-holding)
        # and announces the upcoming death with one DEGRADE_START event. Only
        # reaches a sat that is still NOMINAL -- a sat that already failed in
        # another mode keeps that mode's semantics.
        while self.next_degrade < len(degrades):
            failure = degrades[self.next_degrade]
            if max(failure.t - failure.degrade_s, 0.0) > t + 1e-9:
                break
            self.next_degrade += 1
            i = failure.sat_id
            if status[i] == STATUS_NOMINAL:
                status[i] = STATUS_DEGRADED
                events.append(
                    {
                        "type": "DEGRADE_START",
                        "payload": {
                            "sat_id": i,
                            "t_fail": failure.t,
                            "degrade_s": failure.degrade_s,
                        },
                    }
                )
                # Degrade-trigger policy (GATES.md 2026-10-29 pre-registration):
                # the EPS_DEGRADE fade window is itself the warning, so a
                # critical-role holder that just turned DEGRADED fires the
                # oracle path's proactive critical-seat swap -- no oracle, no
                # predictor, L=degrade_s timing for free. Once per transition
                # (this block only runs on the NOMINAL -> DEGRADED flip),
                # event-driven, fully deterministic. The fading satellite ends
                # up in a non-critical slot but stays NON-assignable (status
                # != NOMINAL -- existing replan rules untouched).
                if (
                    scenario.reassign_on_degrade
                    and scenario.recovery_enabled
                    and roles[i] in _CRITICAL_ROLES
                    and self.mission in (STATE_NOMINAL, STATE_RECOVERED)
                ):
                    failing_soon = {
                        f.sat_id for f in scenario.failures if f.t > t + 1e-9
                    }
                    event = _swap_critical_seat(
                        i,
                        roles[i],
                        failing_soon,
                        [s for s in range(n_sats) if status[s] == STATUS_NOMINAL],
                        roles,
                        seat_of,
                        t,
                        "degrade_trigger",
                    )
                    if event is not None:
                        events.append(event)
        # Failures fire (before this tick's heartbeat), in scheduled order.
        while self.next_failure < len(failures) and failures[self.next_failure].t <= t + 1e-9:
            failure = failures[self.next_failure]
            self.next_failure += 1
            i = failure.sat_id
            if status[i] != STATUS_DEAD:
                status[i] = (
                    STATUS_THRUSTER_OUT
                    if failure.mode == FailureMode.THRUSTER_OUT
                    else STATUS_DEAD
                )
                t_fail[i] = failure.t
                fail_mode[i] = str(failure.mode.value)
                if failure.mode == FailureMode.THRUSTER_OUT:
                    # Self-reported failure (Addendum v1.2): the sat announces
                    # its own dead thruster, so detection needs no heartbeat
                    # misses and fires on this tick (t_detect = t, equal to
                    # t_fail for tick-aligned failure times -- every committed
                    # scenario and live injection) and the machine enters
                    # FAILURE_DETECTED immediately -- the standard REPLANNING
                    # -> re-election -> RECONFIGURING -> RECOVERED arc then
                    # runs unchanged (non-NOMINAL sats are already excluded
                    # from assignable and seat_roles_for re-elects the role).
                    self.mission = STATE_FAILURE_DETECTED
                    if self.t_first_detect is None:
                        self.t_first_detect = t
                    events.append(
                        {
                            "type": "FAILURE_DETECTED",
                            "payload": {
                                "sat_id": i,
                                "mode": fail_mode[i],
                                "t_fail": failure.t,
                                "t_detect": t,
                                "self_report": True,
                            },
                        }
                    )
        # Heartbeats: only DEAD sats stay silent; everyone else reports each tick.
        for i in range(n_sats):
            if status[i] == STATUS_DEAD:
                missed[i] += 1
            else:
                missed[i] = 0
        # Detection: declare after K consecutive missed heartbeats.
        for i in range(n_sats):
            if (
                status[i] == STATUS_DEAD
                and not declared[i]
                and missed[i] >= scenario.detection.missed_ticks_k
            ):
                declared[i] = True
                self.mission = STATE_FAILURE_DETECTED
                if self.t_first_detect is None:
                    self.t_first_detect = t
                events.append(
                    {
                        "type": "FAILURE_DETECTED",
                        "payload": {
                            "sat_id": i,
                            "mode": fail_mode[i],
                            "t_fail": t_fail[i],
                            "t_detect": t,
                        },
                    }
                )

        # Coverage follows the CURRENT role assignment: re-election after a replan
        # reassigns roles[] (imagers can move to other sats), so the imager set is
        # evaluated per tick. With no replanning roles == scenario.roles, which
        # keeps the no-failure path byte-identical to the G3 nominal log.
        centers = np.array(
            [
                states[i][1:3]
                for i in range(n_sats)
                if roles[i] == "imager" and status[i] != STATUS_DEAD
            ]
        )
        coverage = coverage_fraction(centers, grid, r_f)
        positions = np.array([state[:3] for state in states])
        min_sep = min(
            float(np.linalg.norm(positions[i] - positions[j]))
            for i in range(n_sats)
            for j in range(i + 1, n_sats)
        )
        line = {
            "v": LOG_SCHEMA_VERSION,
            "t": t,
            "state": self.mission,
            "sats": [
                {
                    "id": i,
                    "role": roles[i],
                    "status": status[i],
                    "r": states[i][:3].tolist(),
                    "v": states[i][3:].tolist(),
                    "dv_used": controllers[i].dv_used,
                }
                for i in range(n_sats)
            ],
            "coverage": coverage,
            "min_sep": min_sep,
        }
        if events:
            line["event"] = events.pop(0)
        coverage_by_tick.append((t, coverage))
        self.k = k + 1
        if k == n_ticks - 1:
            return line

        # Oracle value-of-warning hook (Phase 2): gated default-off; inert when disabled
        if (
            scenario.oracle.enabled
            and scenario.recovery_enabled
            and scenario.oracle.lead_s > 0
            and not self.replan_due
            and self.mission in (STATE_NOMINAL, STATE_RECOVERED)
        ):
            lead = scenario.oracle.lead_s
            for f_idx, f in enumerate(failures):
                if not oracle_warned[f_idx] and t < f.t:
                    t_warn = max(f.t - lead, 0.0)
                    if t_warn <= t:
                        oracle_warned[f_idx] = True
                        self.mission = STATE_REPLANNING
                        self.replan_due = True
                        events.append(
                            {
                                "type": "ORACLE_ALERT",
                                "payload": {
                                    "sat_id": f.sat_id,
                                    "lead_s": lead,
                                    "t_fail": f.t,
                                    "t_warn": t,
                                },
                            }
                        )
                        break

        # Between-step orchestration (MASTER_PROMPT 3.5): the replanner runs here,
        # between sim steps, one control cycle per machine transition. The decision
        # states FAILURE_DETECTED and REPLANNING therefore each last exactly one
        # tick in the log; RECONFIGURING spans the physical transfers until every
        # survivor settles within tolerance, and RECOVERED holds to the end.
        if (
            self.mission == STATE_FAILURE_DETECTED
            and scenario.recovery_enabled
            and any(s == STATUS_NOMINAL for s in status)
            and (
                self.t_first_detect is not None
                and t >= self.t_first_detect + scenario.recovery_delay_s - 1e-9
            )
        ):
            self.mission = STATE_REPLANNING
            events.append(
                {
                    "type": "STATE",
                    "payload": {"from": STATE_FAILURE_DETECTED, "to": STATE_REPLANNING},
                }
            )
            self.replan_due = True
            self.t_first_detect = None
        elif self.replan_due and self.mission == STATE_REPLANNING:
            assignable = [i for i in range(n_sats) if status[i] == STATUS_NOMINAL]
            obstacles = [states[i] for i in range(n_sats) if status[i] != STATUS_NOMINAL]
            current_slots = [self.formation[seat_of[i]] for i in assignable]
            plan, pairs, cost_total, method, max_transfer, _cost = replan(
                [states[i] for i in assignable],
                [roles[i] for i in assignable],
                [
                    max(scenario.allocation.dv_budget_mps - controllers[i].dv_used, 1e-9)
                    for i in assignable
                ],
                obstacles,
                scenario.roles,
                scenario.allocation,
                scenario.formation.radial_amp_km * 1000.0,
                r_f,
                grid,
                n,
                scenario.a_max_mps2,
                t,
                current_slots=current_slots,
            )
            if method == "do_no_harm_hold":
                pairs = [(idx, idx) for idx in range(len(assignable))]
                max_transfer = 0.0
                for row, col in pairs:
                    sat_id = assignable[row]
                    roles[sat_id] = plan.seat_roles[col]
            else:
                # DEGRADED sats are excluded from assignable (status != NOMINAL) but
                # keep PD-holding their assigned seat (Addendum v1.2, module
                # docstring) -- so when a replan shrinks the formation to the
                # survivors' k seats, each DEGRADED sat keeps its pre-replan seat:
                # its formation entry is appended after plan.slots (ascending sat
                # id, deterministic) and seat_of is re-pointed. Without this the
                # stale seat index either crashes the step loop (index >= k, first
                # hit by the v2 oracle sweep on eps_downlink_6sat) or silently
                # double-books a survivor's seat (eps_leader_6sat, seat 0).
                # Reached only when a replan fires while a sat is DEGRADED: via the
                # oracle hook, or reactively when a DEAD sat's missed-heartbeat
                # detection fires while another sat is mid-fade (mixed failures --
                # the DEGRADED sat then lands in `preserved` below). No committed
                # scenario does either (the EPS_DEGRADE ones are single-failure
                # with the oracle off), so this block never fires on a canonical
                # run and the golden pins stay byte-identical.
                preserved: list[np.ndarray] = []
                for i in range(n_sats):
                    if status[i] == STATUS_DEGRADED:
                        preserved.append(self.formation[seat_of[i]])
                        seat_of[i] = len(plan.slots) + len(preserved) - 1
                self.formation = list(plan.slots) + preserved
                for row, col in pairs:
                    sat_id = assignable[row]
                    seat_of[sat_id] = col
                    roles[sat_id] = plan.seat_roles[col]
                if scenario.oracle.enabled and scenario.oracle.lead_s > 0:
                    lead = scenario.oracle.lead_s
                    failing_soon = {
                        f.sat_id
                        for f in scenario.failures
                        if t <= f.t <= t + lead + 1e-9
                    }
                    for crit_role in ("leader", "downlink"):
                        holders = [s for s in assignable if roles[s] == crit_role]
                        if not holders:
                            continue
                        event = _swap_critical_seat(
                            holders[0],
                            crit_role,
                            failing_soon,
                            assignable,
                            roles,
                            seat_of,
                            t,
                            None,
                        )
                        if event is not None:
                            events.append(event)
            self.recover_tol = max(_RECOVER_TOL_FRAC * max_transfer, 1.0)
            self.assigned = True
            self.replan_due = False
            self.mission = STATE_RECONFIGURING
            events.append(
                {
                    "type": "ASSIGNMENT",
                    "payload": {
                        "assignments": [
                            {
                                "sat_id": assignable[r],
                                "slot_index": seat_of[assignable[r]],
                                "role": roles[assignable[r]],
                            }
                            for r, _ in sorted(pairs)
                        ],
                        "method": method,
                        "cost_total": cost_total,
                        "formation": {
                            "radial_amp_m": plan.radial_amp_m,
                            "phase_offset_rad": plan.phase0_rad,
                        },
                        # Research patch #1: log the bang-bang lower bound per
                        # reconfig, t >= 2*sqrt(d/a_max) for the largest transfer.
                        "transfer_lower_bound_s": bang_bang_time_lower_bound(
                            max_transfer, scenario.a_max_mps2
                        ),
                    },
                }
            )
        elif self.assigned and self.mission == STATE_RECONFIGURING:
            settled = all(
                float(
                    np.linalg.norm(states[i][:3] - propagate(self.formation[seat_of[i]], t, n)[:3])
                )
                <= self.recover_tol
                for i in range(n_sats)
                if status[i] in (STATUS_NOMINAL, STATUS_DEGRADED)
            )
            if settled:
                self.mission = STATE_RECOVERED
                events.append(
                    {
                        "type": "STATE",
                        "payload": {"from": STATE_RECONFIGURING, "to": STATE_RECOVERED},
                    }
                )

        # Evasive collision avoidance (Addendum v1.2): a discrete bounded evasion
        # maneuver, explicitly NOT potential fields (MASTER_PROMPT 2 lock is
        # preserved; min_sep above stays logged every tick). AFTER the
        # replan/settle block, BEFORE stepping: an evasive-capable sat (NOMINAL
        # or DEGRADED -- THRUSTER_OUT cannot thrust, DEAD does not report) with
        # no active episode starts one when its nearest other sat (ANY status --
        # dead sats coast and stay hazards) is inside the evade radius. The burn
        # points straight away from the intruder at a_max for evade_ticks ticks;
        # a sat still inside the radius when its counter expires starts a new
        # episode (continuous burn until clear -- deterministic, index-ordered
        # scan with a strict < so ties resolve to the lowest id). Evasion changes
        # NO mission state -- only physics, dv_used, and events (FIFO, one pops
        # per tick line). Default-inertness is geometric: no committed scenario
        # comes within the 1000 m default radius, so this block appends nothing
        # and the runs stay byte-identical.
        if scenario.safety.evasion_enabled:
            for i in range(n_sats):
                if (
                    status[i] not in (STATUS_NOMINAL, STATUS_DEGRADED)
                    or evade_ticks_left[i] > 0
                ):
                    continue
                best_j = -1
                best_range = scenario.safety.evade_radius_m
                best_d = None
                for j in range(n_sats):
                    if j == i:
                        continue
                    d_ij = states[i][:3] - states[j][:3]
                    rng = float(np.linalg.norm(d_ij))
                    if 0.0 < rng < best_range:  # rng == 0 has no away-direction
                        best_j, best_range, best_d = j, rng, d_ij
                if best_j >= 0 and best_d is not None:
                    u_evade[i] = scenario.a_max_mps2 * best_d / best_range
                    evade_ticks_left[i] = scenario.safety.evade_ticks
                    events.append(
                        {
                            "type": "EVASIVE_MANEUVER",
                            "payload": {
                                "sat_id": i,
                                "other_id": best_j,
                                "range_m": best_range,
                                "t": t,
                            },
                        }
                    )

        # Step every satellite: healthy and DEGRADED sats PD-hold their assigned
        # seat -- unless an evasion episode is active, in which case the bounded
        # away-from-intruder burn replaces seat-holding for the episode (dv_used
        # accrues through PDController.apply_accel either way); DEAD and
        # THRUSTER_OUT sats coast ballistically (never freeze) and remain hazards.
        for i in range(n_sats):
            if status[i] in (STATUS_NOMINAL, STATUS_DEGRADED):
                if evade_ticks_left[i] > 0:
                    states[i] = controllers[i].apply_accel(
                        states[i], u_evade[i], dt, n
                    )
                    evade_ticks_left[i] -= 1
                else:
                    ref = propagate(self.formation[seat_of[i]], t, n)
                    states[i], _ = controllers[i].step(states[i], ref[:3], ref[3:], dt, n)
            else:
                states[i] = propagate(states[i], dt, n)
        return line

    def run(self, cancel_event: threading.Event | None = None) -> Iterator[dict]:
        """Yield [header (k == 0 only), ticks..., SUMMARY]; RunCancelled between ticks."""
        if self.k == 0:
            yield self.header()
        while self.k < self.n_ticks:
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled(f"run cancelled before tick {self.k}")
            yield self.step()
        # Time-averaged coverage over the last orbit (MASTER_PROMPT 3.4).
        window_start = self.scenario.duration_s - 2.0 * math.pi / self.n
        tail = [c for (tick_t, c) in self.coverage_by_tick if tick_t >= window_start - 1e-9]
        yield {
            "v": LOG_SCHEMA_VERSION,
            "t": self.coverage_by_tick[-1][0],
            "state": STATE_SUMMARY,
            "coverage_mean_last_orbit": sum(tail) / len(tail),
        }


def _failure_key(f: Failure) -> tuple[float, int]:
    return (f.t, f.sat_id)


def _degrade_key(f: Failure) -> tuple[float, int]:
    return (max(f.t - f.degrade_s, 0.0), f.sat_id)


def run_scenario(
    scenario: Scenario, cancel_event: threading.Event | None = None
) -> list[dict]:
    """Simulate the mission and return the JSONL lines: [INIT header, tick..., SUMMARY].

    cancel_event is a cooperative-cancellation hook for the API layer's compute
    timeout (cubesat/session.py): checked once between ticks, set -> RunCancelled
    with no partial output. It is never set by any CLI/canonical caller, so the
    check is a pure read that leaves every log byte-identical (AGENTS.md
    invariant 2); no wall clock enters the sim itself (section 9.2)."""
    if scenario.engine == "v2":  # v2 seam (cubesat/sim_v2.py)
        from cubesat.sim_v2 import run_scenario_v2
        return run_scenario_v2(scenario, cancel_event)
    return list(SimV1(scenario).run(cancel_event))
