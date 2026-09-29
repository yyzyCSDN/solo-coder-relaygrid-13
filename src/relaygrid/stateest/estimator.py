"""Feeder state estimation.

Fuses SCADA analog telemetry, switch positions, AMI meter voltages and
topology observability into bus-voltage and branch-flow estimates.

Design notes:
    * Raw inputs are never mutated; the estimator works on an internal
      effective topology derived from the snapshot and switch indications.
    * Every measurement carries a source/quality weight.  Missing data falls
      back to low-weight pseudo measurements (bus demand, radial voltage
      sweep, model switch default); conflicts down-weight the lower-quality
      source and are reported on the result.
    * A connected topology component is observable only when it contains a
      voltage anchor (a SCADA/AMI voltage measurement).  Components without
      one are returned as unobservable regions with ``None`` estimates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from relaygrid.models import FeederSnapshot, SwitchState

# Base trust placed in each data source.
SOURCE_BASE_WEIGHT = {"scada": 0.85, "ami": 0.55, "pseudo": 0.15, "model": 0.3}
QUALITY_WEIGHT = {"good": 1.0, "suspect": 0.6}
# A measurement that loses a conflict keeps this fraction of its weight.
CONFLICT_FACTOR = 0.1
# Conflict thresholds.
VOLTAGE_CONFLICT_PU = 0.02
FLOW_CONFLICT_FRAC = 0.25
PSEUDO_FLOW_CONFLICT_FRAC = 0.5
# Radial-sweep attenuation per transformer-grade edge, applied to indirect
# (flow-propagated) voltage evidence.
PROP_EDGE_FACTOR = 0.9
PROP_PSEUDO_EDGE_FACTOR = 0.45


@dataclass(frozen=True)
class VoltageDatum:
    """A voltage observation (SCADA bus voltage or mapped AMI meter)."""

    point_id: str
    bus_id: str
    kv_ll: float
    source: str = "scada"  # "scada" | "ami"
    quality: str = "good"  # "good" | "suspect" | anything else rejected
    timestamp_s: float = 0.0
    version: int = 0


@dataclass(frozen=True)
class FlowDatum:
    """A branch analog measurement, positive from ``branch.source`` to target."""

    point_id: str
    branch_id: str
    kw: float | None = None
    kvar: float | None = None
    current_a: float | None = None
    source: str = "scada"
    quality: str = "good"
    timestamp_s: float = 0.0
    version: int = 0


@dataclass(frozen=True)
class SwitchDatum:
    """A telemetered switch position indication."""

    point_id: str
    branch_id: str
    state: SwitchState
    source: str = "scada"
    quality: str = "good"
    timestamp_s: float = 0.0
    version: int = 0


@dataclass(frozen=True)
class SourceRef:
    point_id: str
    source: str
    version: int | None
    weight: float


@dataclass(frozen=True)
class BusVoltageEstimate:
    bus_id: str
    kv_ll: float | None
    pu: float | None
    confidence: float
    observable: bool
    sources: tuple[SourceRef, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BranchFlowEstimate:
    branch_id: str
    kw: float | None
    kvar: float | None
    current_a: float | None
    confidence: float
    observable: bool
    sources: tuple[SourceRef, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Conflict:
    kind: str  # "voltage" | "flow" | "switch"
    element_id: str
    point_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class UnobservableRegion:
    buses: tuple[str, ...]
    branches: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class StateEstimate:
    bus_voltages: dict[str, BusVoltageEstimate]
    branch_flows: dict[str, BranchFlowEstimate]
    unobservable_regions: tuple[UnobservableRegion, ...]
    conflicts: tuple[Conflict, ...]
    confidence: float
    versions: dict[str, int]
    as_of_s: float

    def as_dict(self) -> dict:
        """JSON-serialisable summary of the estimate."""

        def refs(rs):
            return [
                {"point_id": r.point_id, "source": r.source,
                 "version": r.version, "weight": round(r.weight, 6)}
                for r in rs
            ]

        return {
            "as_of_s": self.as_of_s,
            "confidence": round(self.confidence, 6),
            "bus_voltages": {
                b: {"kv_ll": e.kv_ll, "pu": e.pu,
                    "confidence": round(e.confidence, 6),
                    "observable": e.observable, "sources": refs(e.sources)}
                for b, e in sorted(self.bus_voltages.items())
            },
            "branch_flows": {
                b: {"kw": e.kw, "kvar": e.kvar,
                    "current_a": e.current_a,
                    "confidence": round(e.confidence, 6),
                    "observable": e.observable, "sources": refs(e.sources)}
                for b, e in sorted(self.branch_flows.items())
            },
            "unobservable_regions": [
                {"buses": list(r.buses), "branches": list(r.branches),
                 "reason": r.reason}
                for r in self.unobservable_regions
            ],
            "conflicts": [
                {"kind": c.kind, "element_id": c.element_id,
                 "point_ids": list(c.point_ids), "detail": c.detail}
                for c in self.conflicts
            ],
            "versions": dict(self.versions),
        }


def _as_state(value) -> SwitchState:
    return value if isinstance(value, SwitchState) else SwitchState(str(value))


def _weight(source: str, quality: str, timestamp_s: float,
            now_s: float, max_age_s: float) -> float:
    base = SOURCE_BASE_WEIGHT.get(source, 0.4)
    q = QUALITY_WEIGHT.get(quality, 0.0)
    if max_age_s > 0:
        age = max(0.0, now_s - timestamp_s)
        freshness = max(0.0, 1.0 - age / max_age_s)
    else:
        freshness = 1.0
    return base * q * freshness


def _combine(weights) -> float:
    """Noisy-OR style combination: agreeing independent evidence accumulates."""
    residual = 1.0
    for w in weights:
        residual *= 1.0 - min(1.0, max(0.0, w))
    return 1.0 - residual


def _robust_fuse(candidates, tol_value: float, conflict_kind: str,
                 element_id: str, conflicts: list[Conflict]):
    """Weighted mean that down-weights sources conflicting with the strongest.

    ``candidates`` is a list of ``[value, weight, SourceRef]`` lists, mutated
    in place when a member loses a conflict.  Returns
    ``(value, raw_weight_sum, refs_with_final_weight)`` or ``None``.
    """
    active = [c for c in candidates if c[1] > 0.0]
    if not active:
        return None, 0.0, []

    already: set[int] = set()
    while True:
        anchor = min(active, key=lambda c: (-c[1], c[2].point_id))
        flagged = []
        for cand in active:
            if cand is anchor or cand[1] <= 0.0 or id(cand) in already:
                continue
            if abs(cand[0] - anchor[0]) > tol_value and cand[1] < anchor[1] + 1e-12:
                cand[1] *= CONFLICT_FACTOR
                flagged.append(cand)
                already.add(id(cand))
        if not flagged:
            break
        active = [c for c in active if c[1] > 0.0]
        ids = tuple(sorted({anchor[2].point_id} | {c[2].point_id for c in flagged}))
        conflicts.append(Conflict(
            conflict_kind, element_id, ids,
            f"{len(flagged)} lower-quality reading(s) down-weighted versus "
            f"{anchor[2].point_id}"))

    wsum = sum(c[1] for c in active)
    value = sum(c[0] * c[1] for c in active) / wsum
    refs = tuple(
        SourceRef(c[2].point_id, c[2].source, c[2].version, c[1])
        for c in sorted(active, key=lambda c: c[2].point_id)
    )
    return value, wsum, refs


def _resolve_switches(snapshot: FeederSnapshot, switches, now_s, max_age_s,
                      conflicts):
    """Return effective state + supporting refs per switchable branch."""
    effective: dict[str, tuple[SwitchState, list[SourceRef]]] = {}
    groups: dict[str, dict[SwitchState, list] ] = {}

    for d in switches:
        if d.branch_id not in snapshot.branches:
            raise ValueError(f"switch datum references unknown branch {d.branch_id}")
        br = snapshot.branches[d.branch_id]
        if not br.switchable:
            conflicts.append(Conflict(
                "switch", d.branch_id, (d.point_id,),
                "position reported for non-switchable branch; ignored"))
            continue
        w = _weight(d.source, d.quality, d.timestamp_s, now_s, max_age_s)
        if w <= 0.0:
            continue
        groups.setdefault(d.branch_id, {}).setdefault(
            _as_state(d.state), []).append(
            [w, d.point_id, d.version, d.source])

    for br in snapshot.branches.values():
        if not br.switchable:
            continue
        default = snapshot.switch_state.get(br.branch_id, SwitchState.CLOSED)
        if br.branch_id not in groups:
            ref = SourceRef(f"model:{br.branch_id}", "model", None,
                            SOURCE_BASE_WEIGHT["model"])
            effective[br.branch_id] = (default, [ref])
            continue
        by_state = groups[br.branch_id]
        # Winner: total source weight, then newest version, then state name.
        ranked = sorted(
            by_state.items(),
            key=lambda kv: (-sum(x[0] for x in kv[1]),
                            -max(x[2] for x in kv[1]), kv[0].value))
        winner_state, winner_rows = ranked[0]
        if len(by_state) > 1:
            ids = tuple(sorted(x[1] for rows in by_state.values() for x in rows))
            conflicts.append(Conflict(
                "switch", br.branch_id, ids,
                f"position disagreement; adopted '{winner_state.value}'"))
        refs = [
            SourceRef(pid, src, ver, w)
            for w, pid, ver, src in
            sorted(winner_rows, key=lambda x: x[1])
        ]
        effective[br.branch_id] = (winner_state, refs)
    return effective


def estimate_feeder_state(
    snapshot: FeederSnapshot,
    voltages=(),
    flows=(),
    switches=(),
    *,
    now_s: float | None = None,
    max_age_s: float = 120.0,
    reference_bus: str | None = None,
):
    """Estimate bus voltages and branch flows from heterogeneous measurements.

    Parameters mirror the four input classes: SCADA analog measurements
    (voltage/flow), switch position indications, AMI voltages (supplied as
    :class:`VoltageDatum` with ``source="ami"``, already mapped meter->bus)
    and topology observability derived from ``snapshot``.

    Returns a :class:`StateEstimate`; ``snapshot`` and all inputs are left
    unchanged.
    """
    if max_age_s < 0:
        raise ValueError("max age cannot be negative")
    stamps = [d.timestamp_s for d in (*voltages, *flows, *switches)]
    as_of = float(now_s) if now_s is not None else (max(stamps) if stamps else 0.0)

    for d in voltages:
        if d.bus_id not in snapshot.buses:
            raise ValueError(f"voltage datum references unknown bus {d.bus_id}")
    for d in flows:
        if d.branch_id not in snapshot.branches:
            raise ValueError(f"flow datum references unknown branch {d.branch_id}")
    if reference_bus is not None and reference_bus not in snapshot.buses:
        raise ValueError(f"reference bus {reference_bus} not in snapshot")

    conflicts: list[Conflict] = []
    versions: dict[str, int] = {}

    # --- effective topology from switch positions ----------------------
    switches_eff = _resolve_switches(snapshot, switches, as_of, max_age_s,
                                     conflicts)

    def is_closed(br) -> bool:
        if not br.switchable:
            return True
        return switches_eff[br.branch_id][0] == SwitchState.CLOSED

    open_refs: dict[str, list[SourceRef]] = {}
    for br_id, (state, refs) in switches_eff.items():
        if state == SwitchState.OPEN:
            open_refs[br_id] = refs
        for r in refs:
            if r.version is not None:
                versions[r.point_id] = r.version

    adj: dict[str, list[tuple]] = {b: [] for b in snapshot.buses}
    closed_branches: list = []
    for br in snapshot.branches.values():
        if is_closed(br):
            closed_branches.append(br)
            adj[br.source].append((br, br.target))
            adj[br.target].append((br, br.source))

    # --- connected components of the energized graph --------------------
    components: list[set[str]] = []
    seen: set[str] = set()
    for start in sorted(adj):
        if start in seen:
            continue
        comp = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(nb for _, nb in adj[cur] if nb not in comp)
        seen |= comp
        components.append(comp)

    # --- direct voltage observations per bus ----------------------------
    vobs: dict[str, list] = {}
    for d in voltages:
        w = _weight(d.source, d.quality, d.timestamp_s, as_of, max_age_s)
        if w <= 0.0:
            continue
        vobs.setdefault(d.bus_id, []).append(
            [float(d.kv_ll), w, SourceRef(d.point_id, d.source, d.version, w)])

    direct_v: dict[str, tuple[float, float, tuple]] = {}
    direct_w: dict[str, float] = {}
    for bus_id, cands in vobs.items():
        nominal = snapshot.buses[bus_id].kv_ll
        value, wsum, refs = _robust_fuse(
            cands, VOLTAGE_CONFLICT_PU * nominal, "voltage", bus_id, conflicts)
        if value is not None:
            direct_v[bus_id] = (value, wsum, refs)
            direct_w[bus_id] = _combine(r.weight for r in refs)
            for r in refs:
                versions[r.point_id] = r.version

    # --- flow observations per branch -----------------------------------
    fobs: dict[str, list] = {}
    for d in flows:
        w = _weight(d.source, d.quality, d.timestamp_s, as_of, max_age_s)
        if w <= 0.0:
            continue
        br = snapshot.branches[d.branch_id]
        nominal = (snapshot.buses[br.source].kv_ll
                   if br.source in snapshot.buses else 0.0)
        kw, kvar = float(d.kw) if d.kw is not None else None, (
            float(d.kvar) if d.kvar is not None else None)
        if d.current_a is not None and (kw is None or kvar is None):
            kva = math.sqrt(3.0) * nominal * float(d.current_a)
            # Split apparent power on the pseudo downstream power factor.
            p0 = snapshot.buses[br.target].demand_kw if br.target in snapshot.buses else 0.0
            q0 = snapshot.buses[br.target].demand_kvar if br.target in snapshot.buses else 0.0
            s0 = math.hypot(p0, q0)
            if s0 > 0:
                kw, kvar = kva * p0 / s0, kva * q0 / s0
            else:
                kw, kvar = kva, 0.0
        if kw is None:
            continue
        fobs.setdefault(d.branch_id, []).append(
            [kw, kvar if kvar is not None else 0.0, w,
             SourceRef(d.point_id, d.source, d.version, w)])

    # --- estimate component by component --------------------------------
    bus_v: dict[str, float] = {}
    bus_w: dict[str, float] = {}
    bus_refs: dict[str, list[SourceRef]] = {}
    branch_est: dict[str, tuple[float, float, float, float, list]] = {}
    unobservable: list[UnobservableRegion] = []

    for comp in components:
        anchored = sorted(b for b in direct_v if b in comp)
        if not anchored:
            internal = sorted(br.branch_id for br in closed_branches
                              if br.source in comp and br.target in comp)
            boundary = sorted(
                br.branch_id for br in snapshot.branches.values()
                if (br.source in comp) ^ (br.target in comp))
            unobservable.append(UnobservableRegion(
                tuple(sorted(comp)), tuple(internal + boundary),
                "no_voltage_anchor"))
            continue

        if reference_bus in comp:
            root = reference_bus
        else:
            root = sorted(comp, key=lambda b: (-direct_w.get(b, 0.0), b))[0]

        # BFS tree oriented out of the anchor.
        parent: dict[str, tuple] = {root: None}
        order = [root]
        for cur in order:
            for br, nb in sorted(adj[cur], key=lambda e: e[0].branch_id):
                if nb not in parent:
                    parent[nb] = (br, cur)
                    order.append(nb)

        # Backward sweep: downstream pseudo demand (low-weight fallback).
        down_p = {b: snapshot.buses[b].demand_kw for b in comp}
        down_q = {b: snapshot.buses[b].demand_kvar for b in comp}
        pseudo_flow: dict[str, tuple[float, float]] = {}
        for child in reversed(order[1:]):
            br, par = parent[child]
            down_p[par] += down_p[child]
            down_q[par] += down_q[child]
            pseudo_flow[br.branch_id] = (down_p[child], down_q[child])

        # Root voltage from direct evidence.
        bus_v[root] = direct_v[root][0]
        bus_w[root] = direct_w[root]
        bus_refs[root] = list(direct_v[root][2])
        anchor_w = direct_w[root]
        prop_w = {root: anchor_w}

        # Forward sweep with per-branch flow fusion.
        depth = {root: 0}
        for child in order[1:]:
            br, par = parent[child]
            p0, q0 = pseudo_flow[br.branch_id]
            reverse = child == br.source  # measured positive source->target
            measured = []
            for kw, kvar, w, ref in fobs.get(br.branch_id, []):
                oriented = (-kw, -kvar) if reverse else (kw, kvar)
                measured.append([oriented[0], oriented[1], w, ref])

            sources: list[SourceRef] = []
            edge_measured = bool(measured)

            # Fuse multiple analog readings against each other.
            if measured:
                kw_cands = [[m[0], m[2], m[3]] for m in measured]
                q_cands = [[m[1], m[2], m[3]] for m in measured]
                scale = max(abs(kw_cands[0][0]), 1.0)
                kw_m, kw_w, kw_refs = _robust_fuse(
                    kw_cands, FLOW_CONFLICT_FRAC * scale, "flow",
                    br.branch_id, conflicts)
                kvar_m, _, q_refs = _robust_fuse(
                    q_cands, FLOW_CONFLICT_FRAC * max(abs(q_cands[0][0]), 1.0),
                    "flow", br.branch_id, conflicts)
                # Down-weight the pseudo demand when it contradicts telemetry.
                pseudo_w = SOURCE_BASE_WEIGHT["pseudo"]
                if abs(p0 - kw_m) > PSEUDO_FLOW_CONFLICT_FRAC * max(abs(kw_m), 1.0):
                    pseudo_w *= CONFLICT_FACTOR
                    conflicts.append(Conflict(
                        "flow", br.branch_id,
                        tuple(sorted(r.point_id for r in kw_refs)),
                        "downstream pseudo demand disagrees with telemetry; "
                        "pseudo flow down-weighted"))
                p_o = (kw_m * kw_w + p0 * pseudo_w) / (kw_w + pseudo_w)
                q_o = (kvar_m * kw_w + q0 * pseudo_w) / (kw_w + pseudo_w)
                for r in kw_refs:
                    versions[r.point_id] = r.version
                    sources.append(r)
            else:
                p_o, q_o, pseudo_w = p0, q0, SOURCE_BASE_WEIGHT["pseudo"]

            pseudo_ref = SourceRef(f"demand:{br.branch_id}", "pseudo", None,
                                   pseudo_w)
            sources.append(pseudo_ref)

            # Report flow in the branch's source->target convention.
            p_out, q_out = (-p_o, -q_o) if reverse else (p_o, q_o)
            v_send = bus_v[par]
            current = math.hypot(p_out, q_out) / (
                math.sqrt(3.0) * max(v_send, 1e-9))
            flow_w = min(1.0, sum(r.weight for r in sources
                                  if r.source != "pseudo")) if edge_measured else 0.0
            branch_est[br.branch_id] = (p_out, q_out, current, flow_w, sources)

            # Indirect voltage via radial drop from the parent estimate.
            drop_kv = (br.r_ohm * p_o + br.x_ohm * q_o) / (v_send * 1000.0)
            prop_v = v_send - drop_kv
            depth[child] = depth[par] + 1
            factor = PROP_EDGE_FACTOR * (1.0 if edge_measured
                                         else PROP_PSEUDO_EDGE_FACTOR)
            prop_w[child] = prop_w[par] * factor

            cands_v = []
            refs_v = []
            if child in direct_v:
                value, _, refs = direct_v[child]
                cands_v.append((value, direct_w[child]))
                refs_v.extend(refs)
            cands_v.append((prop_v, prop_w[child]))
            refs_v.append(SourceRef(f"sweep:{br.branch_id}", "model", None,
                                    prop_w[child]))
            wsum_v = sum(w for _, w in cands_v)
            bus_v[child] = sum(v * w for v, w in cands_v) / wsum_v
            bus_w[child] = _combine([w for _, w in cands_v])
            bus_refs[child] = refs_v

    # --- assemble outputs ------------------------------------------------
    bus_results: dict[str, BusVoltageEstimate] = {}
    for bus_id, bus in snapshot.buses.items():
        if bus_id in bus_v:
            v = bus_v[bus_id]
            bus_results[bus_id] = BusVoltageEstimate(
                bus_id, v, v / bus.kv_ll if bus.kv_ll > 0 else None,
                bus_w[bus_id], True, tuple(bus_refs[bus_id]))
        else:
            bus_results[bus_id] = BusVoltageEstimate(
                bus_id, None, None, 0.0, False, ())

    branch_results: dict[str, BranchFlowEstimate] = {}
    for br in snapshot.branches.values():
        if br.branch_id in branch_est:
            p_out, q_out, current, flow_w, sources = branch_est[br.branch_id]
            direct_endpoints = int(br.source in direct_w) + int(
                br.target in direct_w)
            terms = [flow_w] if flow_w > 0 else [
                SOURCE_BASE_WEIGHT["pseudo"]]
            if direct_endpoints == 2:
                terms.append(0.4)
            branch_results[br.branch_id] = BranchFlowEstimate(
                br.branch_id, p_out, q_out, current,
                _combine(terms), True, tuple(sources))
        elif br.branch_id in open_refs:  # open switch: zero flow by topology
            refs = tuple(open_refs[br.branch_id])
            branch_results[br.branch_id] = BranchFlowEstimate(
                br.branch_id, 0.0, 0.0, 0.0,
                min(1.0, sum(r.weight for r in refs)), True, refs)
            for r in refs:
                if r.version is not None:
                    versions[r.point_id] = r.version
        else:
            branch_results[br.branch_id] = BranchFlowEstimate(
                br.branch_id, None, None, None, 0.0, False, ())

    scoreables = [e.confidence for e in bus_results.values()]
    scoreables += [e.confidence for e in branch_results.values()]
    overall = sum(scoreables) / len(scoreables) if scoreables else 0.0

    return StateEstimate(
        bus_results, branch_results, tuple(unobservable),
        tuple(conflicts), overall, dict(sorted(versions.items())), as_of)
