"""Feeder state estimation.

Merges telemetry analog measurements, switch positions (topology), AMI
voltage, and topology observability into bus voltage and branch power flow
estimates.

Raw inputs are never mutated. Missing and conflicting data are down-weighted
by source quality. The result carries per-estimate confidence, the set of
unobservable areas, and the data version used.
"""
from __future__ import annotations

from dataclasses import dataclass

from relaygrid.models import FeederSnapshot

# ---------------------------------------------------------------------------
# Source quality weights (higher = more trustworthy).
# Mirrors the SOURCE_WEIGHT convention used in outage reconciliation.
# ---------------------------------------------------------------------------
SOURCE_WEIGHT: dict[str, float] = {
    "telemetry": 1.0,  # SCADA analog measurement
    "ami": 0.85,  # AMI voltage measurement
    "pseudo": 0.4,  # pseudo-measurement derived from known load
    "nominal": 0.5,  # nominal / assumed value
}

QUALITY_FACTOR: dict[str, float] = {
    "good": 1.0,
    "suspect": 0.6,
    "bad": 0.0,
}

# Confidence below which a bus / branch is reported as unobservable.
UNOBSERVABLE_THRESHOLD = 0.3

# Measurements disagreeing by more than this relative fraction conflict.
CONFLICT_THRESHOLD = 0.05

# Propagation parameters for inferred (non-measured) estimates.
# Confidence decays with each branch traversed from the nearest measurement.
INFERRED_BASE = 0.8
INFERRED_DECAY = 0.95

# Confidence assigned to the source bus when only nominal voltage is known.
SOURCE_NOMINAL_CONFIDENCE = 0.7


# ---------------------------------------------------------------------------
# Input measurement types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoltageMeasurement:
    """A line-to-line voltage measurement at a bus (kV)."""

    bus_id: str
    value_kv: float
    source: str = "telemetry"
    quality: str = "good"
    version: int = 0


@dataclass(frozen=True)
class FlowMeasurement:
    """A branch power flow measurement (kW, kvar)."""

    branch_id: str
    kw: float
    kvar: float
    source: str = "telemetry"
    quality: str = "good"
    version: int = 0


# ---------------------------------------------------------------------------
# Output estimate types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BusEstimate:
    bus_id: str
    kv: float
    confidence: float
    observable: bool
    source: str  # "measurement" | "inferred" | "nominal" | "unobservable"


@dataclass(frozen=True)
class BranchFlowEstimate:
    branch_id: str
    kw: float
    kvar: float
    confidence: float
    observable: bool
    source: str


@dataclass(frozen=True)
class FeederStateEstimate:
    buses: dict[str, BusEstimate]
    branches: dict[str, BranchFlowEstimate]
    unobservable_areas: list[frozenset[str]]
    data_version: int
    measurement_count: int
    conflict_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _weight(source: str, quality: str) -> float:
    return SOURCE_WEIGHT.get(source, 0.2) * QUALITY_FACTOR.get(quality, 0.0)


def _build_adjacency(snapshot: FeederSnapshot) -> dict[str, list[tuple[str, str]]]:
    """Bus -> [(neighbor, branch_id)] over closed branches only."""
    adj: dict[str, list[tuple[str, str]]] = {b: [] for b in snapshot.buses}
    for br in snapshot.closed_branches():
        adj[br.source].append((br.target, br.branch_id))
        adj[br.target].append((br.source, br.branch_id))
    return adj


def _energized_buses(adj: dict[str, list[tuple[str, str]]], source_bus: str) -> set[str]:
    """Buses reachable from the source through closed branches."""
    seen: set[str] = set()
    stack = [source_bus]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(n for n, _ in adj[cur] if n not in seen)
    return seen


def _bfs_order(adj: dict[str, list[tuple[str, str]]], source_bus: str) -> list[str]:
    """Buses in BFS order from the source (parent before child)."""
    order: list[str] = []
    seen: set[str] = set()
    queue = [source_bus]
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        order.append(cur)
        queue.extend(n for n, _ in adj[cur] if n not in seen)
    return order


def _branch_between(adj: dict[str, list[tuple[str, str]]], a: str, b: str) -> str | None:
    for n, bid in adj[a]:
        if n == b:
            return bid
    return None


def _blend_voltages(measurements: list[VoltageMeasurement]) -> tuple[float, float, bool]:
    """Weighted mean of voltage measurements with conflict down-weighting.

    When measurements conflict, the lowest-quality measurement (and among
    ties the one farthest from the consensus) is down-weighted. Returns
    (kv, confidence, conflict_detected).
    """
    weights = [_weight(m.source, m.quality) for m in measurements]
    values = [m.value_kv for m in measurements]
    conflict = False
    if len(values) > 1:
        total_w = sum(weights)
        mean_val = sum(v * w for v, w in zip(values, weights)) / total_w
        # Target the lowest-quality measurement; among ties, the largest
        # deviation from consensus.
        min_w = min(weights)
        candidates = [i for i, w in enumerate(weights) if w == min_w]
        worst = max(candidates, key=lambda i: abs(values[i] - mean_val))
        if abs(values[worst] - mean_val) > CONFLICT_THRESHOLD * max(abs(mean_val), 1e-9):
            weights[worst] *= 0.5
            conflict = True
    total_w = sum(weights)
    est = sum(v * w for v, w in zip(values, weights)) / total_w if total_w > 0 else 0.0
    conf = min(1.0, total_w / len(weights)) if weights else 0.0
    return est, conf, conflict


def _connected_groups(buses: set[str], adj: dict[str, list[tuple[str, str]]]) -> list[frozenset[str]]:
    """Connected groups of the given buses over the adjacency."""
    groups: list[frozenset[str]] = []
    seen: set[str] = set()
    for start in sorted(buses):
        if start in seen:
            continue
        group: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in buses:
                continue
            seen.add(cur)
            group.add(cur)
            stack.extend(n for n, _ in adj[cur])
        if group:
            groups.append(frozenset(group))
    return groups


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------
def estimate_feeder_state(
    snapshot: FeederSnapshot,
    *,
    voltage_measurements: list[VoltageMeasurement] | None = None,
    flow_measurements: list[FlowMeasurement] | None = None,
    source_bus: str | None = None,
    nominal_kv: float | None = None,
    data_version: int = 0,
) -> FeederStateEstimate:
    """Estimate bus voltages and branch flows for a feeder.

    Merges telemetry analog, AMI voltage, switch positions (topology), and
    topology observability. Raw inputs are not mutated. Missing/conflicting
    data are down-weighted by source quality.

    Returns bus estimates, branch flow estimates, unobservable areas, the
    data version used, and conflict counts.
    """
    vmeas = voltage_measurements or []
    fmeas = flow_measurements or []

    # --- Resolve source and nominal voltage ---
    if source_bus is None:
        source_bus = next(iter(snapshot.buses), None)
    if source_bus is None or source_bus not in snapshot.buses:
        raise ValueError("source bus not found in snapshot")
    if nominal_kv is None:
        nominal_kv = snapshot.buses[source_bus].kv_ll

    # --- Energized topology ---
    adj = _build_adjacency(snapshot)
    energized = _energized_buses(adj, source_bus)

    # --- Assign measurements (drop bad quality) ---
    v_by_bus: dict[str, list[VoltageMeasurement]] = {}
    for m in vmeas:
        if m.quality == "bad":
            continue
        v_by_bus.setdefault(m.bus_id, []).append(m)

    f_by_branch: dict[str, list[FlowMeasurement]] = {}
    for m in fmeas:
        if m.quality == "bad":
            continue
        f_by_branch.setdefault(m.branch_id, []).append(m)

    measurement_count = sum(len(v) for v in v_by_bus.values()) + sum(
        len(f) for f in f_by_branch.values()
    )

    # --- Rooted tree (parent / children) via BFS ---
    order = _bfs_order(adj, source_bus)
    parent_map: dict[str, str | None] = {source_bus: None}
    for parent in order:
        for n, _ in adj[parent]:
            if n not in parent_map:
                parent_map[n] = parent

    # --- Backward sweep: branch flows from loads ---
    downstream_kw: dict[str, float] = {b: 0.0 for b in snapshot.buses}
    downstream_kvar: dict[str, float] = {b: 0.0 for b in snapshot.buses}
    for b in snapshot.buses:
        if b in energized:
            downstream_kw[b] = snapshot.buses[b].demand_kw
            downstream_kvar[b] = snapshot.buses[b].demand_kvar

    branch_flow_est: dict[str, tuple[float, float, float, str]] = {}
    conflict_count = 0

    for bus in reversed(order):
        if bus == source_bus:
            continue
        parent = parent_map[bus]
        if parent is None:
            continue
        branch_id = _branch_between(adj, parent, bus)
        if branch_id is None:
            continue

        # Flow through this branch = downstream demand of this bus.
        kw = downstream_kw[bus]
        kvar = downstream_kvar[bus]
        downstream_kw[parent] += kw
        downstream_kvar[parent] += kvar

        direct = f_by_branch.get(branch_id, [])
        if direct:
            weights = [_weight(m.source, m.quality) for m in direct]
            total_w = sum(weights)
            dkw = sum(m.kw * w for m, w in zip(direct, weights)) / total_w
            dkvar = sum(m.kvar * w for m, w in zip(direct, weights)) / total_w
            if abs(dkw - kw) > CONFLICT_THRESHOLD * max(abs(dkw), 1e-9) or abs(dkvar - kvar) > CONFLICT_THRESHOLD * max(abs(dkvar), 1e-9):
                conflict_count += 1
                kw = 0.5 * kw + 0.5 * dkw
                kvar = 0.5 * kvar + 0.5 * dkvar
            else:
                kw = dkw
                kvar = dkvar
            conf = min(1.0, total_w / len(direct))
            branch_flow_est[branch_id] = (kw, kvar, conf, "measurement")
        else:
            has_load = downstream_kw[bus] > 0 or downstream_kvar[bus] > 0
            conf = 0.6 if has_load else 0.4
            branch_flow_est[branch_id] = (kw, kvar, conf, "inferred")

    # --- Forward sweep: bus voltages from source ---
    bus_est: dict[str, BusEstimate] = {}

    if source_bus in v_by_bus:
        kv, conf, _ = _blend_voltages(v_by_bus[source_bus])
        src = "measurement"
    else:
        kv = nominal_kv
        conf = SOURCE_NOMINAL_CONFIDENCE
        src = "nominal"
    bus_est[source_bus] = BusEstimate(source_bus, kv, conf, conf >= UNOBSERVABLE_THRESHOLD, src)

    for bus in order:
        if bus == source_bus:
            continue
        parent = parent_map[bus]
        if parent is None:
            continue
        branch_id = _branch_between(adj, parent, bus)
        if branch_id is None:
            continue
        br = snapshot.branches[branch_id]
        parent_est = bus_est[parent]

        flow = branch_flow_est.get(branch_id)
        if flow and parent_est.observable and bus in energized:
            kw, kvar, _, _ = flow
            drop = (br.r_ohm * kw + br.x_ohm * kvar) / max(parent_est.kv, 1e-9) / 1000.0
            inferred_kv = max(0.0, parent_est.kv - drop)
            inferred_conf = min(INFERRED_BASE, parent_est.confidence * INFERRED_DECAY)
        else:
            inferred_kv = parent_est.kv
            inferred_conf = 0.0

        if bus in v_by_bus:
            mkv, mconf, conflict = _blend_voltages(v_by_bus[bus])
            if conflict:
                conflict_count += 1
            if inferred_conf > 0:
                total_c = mconf + inferred_conf
                kv = (mconf * mkv + inferred_conf * inferred_kv) / total_c
                conf = min(1.0, (mconf + inferred_conf) / 2.0)
            else:
                kv = mkv
                conf = mconf
            src = "measurement"
        else:
            kv = inferred_kv
            conf = inferred_conf
            src = "inferred" if conf > 0 else "unobservable"

        observable = conf >= UNOBSERVABLE_THRESHOLD and bus in energized
        bus_est[bus] = BusEstimate(bus, kv, conf, observable, src)

    # De-energized buses are unobservable.
    for bus in snapshot.buses:
        if bus not in energized:
            bus_est[bus] = BusEstimate(bus, 0.0, 0.0, False, "unobservable")

    # --- Branch flow estimates for all branches ---
    branch_est: dict[str, BranchFlowEstimate] = {}
    for br_id, br in snapshot.branches.items():
        if br_id in branch_flow_est:
            kw, kvar, conf, src = branch_flow_est[br_id]
        else:
            kw, kvar, conf, src = 0.0, 0.0, 0.0, "unobservable"
        observable = (
            conf >= UNOBSERVABLE_THRESHOLD
            and br.source in energized
            and br.target in energized
        )
        branch_est[br_id] = BranchFlowEstimate(br_id, kw, kvar, conf, observable, src)

    # --- Unobservable areas ---
    unobs_buses = {b for b, e in bus_est.items() if not e.observable}
    unobs_areas = _connected_groups(unobs_buses, adj)

    # --- Data version ---
    max_ver = max(
        [m.version for m in vmeas] + [m.version for m in fmeas] + [0]
    )
    out_version = max(data_version, max_ver)

    return FeederStateEstimate(
        buses=bus_est,
        branches=branch_est,
        unobservable_areas=unobs_areas,
        data_version=out_version,
        measurement_count=measurement_count,
        conflict_count=conflict_count,
    )
