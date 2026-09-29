from __future__ import annotations

import json

import pytest

from relaygrid.cli import main
from relaygrid.models import Branch, Bus, FeederSnapshot, SwitchState
from relaygrid.state_estimation import (
    BranchFlowEstimate,
    BusEstimate,
    FeederStateEstimate,
    FlowMeasurement,
    VoltageMeasurement,
    estimate_feeder_state,
)


def _feeder(load_c: float = 100.0, load_d: float = 50.0) -> FeederSnapshot:
    """A radial feeder A-B-C-D with all switches closed."""
    buses = {
        "A": Bus("A", 12.47),
        "B": Bus("B", 12.47),
        "C": Bus("C", 12.47, demand_kw=load_c),
        "D": Bus("D", 12.47, demand_kw=load_d),
    }
    branches = {
        "ab": Branch("ab", "A", "B", 0.1, 0.1, 300.0),
        "bc": Branch("bc", "B", "C", 0.1, 0.1, 300.0),
        "cd": Branch("cd", "C", "D", 0.1, 0.1, 300.0),
    }
    return FeederSnapshot(buses, branches, {})


def _feeder_with_tie() -> FeederSnapshot:
    """A-B-C-D with cd open; D is de-energized."""
    buses = {
        "A": Bus("A", 12.47),
        "B": Bus("B", 12.47),
        "C": Bus("C", 12.47, demand_kw=100.0),
        "D": Bus("D", 12.47, demand_kw=50.0),
    }
    branches = {
        "ab": Branch("ab", "A", "B", 0.1, 0.1, 300.0),
        "bc": Branch("bc", "B", "C", 0.1, 0.1, 300.0),
        "cd": Branch("cd", "C", "D", 0.1, 0.1, 300.0, True, True),
    }
    return FeederSnapshot(buses, branches, {"cd": SwitchState.OPEN})


# ---------------------------------------------------------------------------
# Basic estimation
# ---------------------------------------------------------------------------
def test_estimate_returns_all_buses_and_branches():
    snap = _feeder()
    result = estimate_feeder_state(snap)
    assert isinstance(result, FeederStateEstimate)
    assert set(result.buses) == {"A", "B", "C", "D"}
    assert set(result.branches) == {"ab", "bc", "cd"}


def test_source_bus_uses_nominal_when_unmeasured():
    snap = _feeder()
    result = estimate_feeder_state(snap)
    src = result.buses["A"]
    assert src.source == "nominal"
    assert src.kv == pytest.approx(12.47)
    assert src.observable is True
    assert src.confidence > 0.0


def test_measured_bus_has_higher_confidence_than_inferred():
    snap = _feeder()
    vmeas = [VoltageMeasurement("C", 12.45, source="telemetry", quality="good")]
    result = estimate_feeder_state(snap, voltage_measurements=vmeas)
    c = result.buses["C"]
    b = result.buses["B"]
    assert c.source == "measurement"
    assert c.confidence > b.confidence
    # Estimate is a blend of measurement and model-based inference.
    assert abs(c.kv - 12.45) < 0.02


def test_ami_voltage_is_accepted():
    snap = _feeder()
    vmeas = [VoltageMeasurement("C", 12.44, source="ami", quality="good")]
    result = estimate_feeder_state(snap, voltage_measurements=vmeas)
    c = result.buses["C"]
    assert c.source == "measurement"
    assert c.confidence > 0.0
    assert abs(c.kv - 12.44) < 0.02


def test_suspect_quality_lowers_confidence():
    snap = _feeder()
    good = [VoltageMeasurement("C", 12.45, source="telemetry", quality="good")]
    suspect = [VoltageMeasurement("C", 12.45, source="telemetry", quality="suspect")]
    r_good = estimate_feeder_state(snap, voltage_measurements=good)
    r_suspect = estimate_feeder_state(snap, voltage_measurements=suspect)
    assert r_good.buses["C"].confidence > r_suspect.buses["C"].confidence


def test_bad_quality_is_dropped():
    snap = _feeder()
    vmeas = [VoltageMeasurement("C", 12.45, source="telemetry", quality="bad")]
    result = estimate_feeder_state(snap, voltage_measurements=vmeas)
    assert result.buses["C"].source == "inferred"
    assert result.measurement_count == 0


# ---------------------------------------------------------------------------
# Missing data / inference
# ---------------------------------------------------------------------------
def test_no_measurements_still_estimates_observable_buses():
    snap = _feeder()
    result = estimate_feeder_state(snap)
    for bus in result.buses.values():
        assert bus.observable is True
        assert bus.confidence > 0.0


def test_inferred_voltage_decreases_along_feeder():
    snap = _feeder(load_c=100.0, load_d=50.0)
    result = estimate_feeder_state(snap)
    # Voltage drops as we move away from the source.
    assert result.buses["A"].kv >= result.buses["B"].kv
    assert result.buses["B"].kv >= result.buses["C"].kv
    assert result.buses["C"].kv >= result.buses["D"].kv


def test_inferred_confidence_decreases_along_feeder():
    snap = _feeder()
    result = estimate_feeder_state(snap)
    assert result.buses["A"].confidence >= result.buses["B"].confidence
    assert result.buses["B"].confidence >= result.buses["C"].confidence


# ---------------------------------------------------------------------------
# Flow measurements
# ---------------------------------------------------------------------------
def test_flow_measurement_is_used():
    snap = _feeder()
    fmeas = [FlowMeasurement("cd", 50.0, 0.0, source="telemetry", quality="good")]
    result = estimate_feeder_state(snap, flow_measurements=fmeas)
    cd = result.branches["cd"]
    assert cd.source == "measurement"
    assert cd.kw == pytest.approx(50.0)
    assert cd.observable is True


def test_computed_flow_from_loads():
    snap = _feeder(load_c=100.0, load_d=50.0)
    result = estimate_feeder_state(snap)
    # cd carries D's load; bc carries C+D; ab carries all.
    assert result.branches["cd"].kw == pytest.approx(50.0)
    assert result.branches["bc"].kw == pytest.approx(150.0)
    assert result.branches["ab"].kw == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Conflict handling
# ---------------------------------------------------------------------------
def test_conflicting_voltages_are_downweighted():
    snap = _feeder()
    vmeas = [
        VoltageMeasurement("C", 12.45, source="telemetry", quality="good"),
        VoltageMeasurement("C", 11.0, source="ami", quality="good"),  # conflict
    ]
    result = estimate_feeder_state(snap, voltage_measurements=vmeas)
    assert result.conflict_count >= 1
    # The estimate should be between the two values, not equal to the bad one.
    c = result.buses["C"]
    assert 11.0 < c.kv < 12.45


def test_conflicting_flow_measurement_blended():
    snap = _feeder(load_d=50.0)
    # cd computed flow is 50; direct measurement says 200 (conflict).
    fmeas = [FlowMeasurement("cd", 200.0, 0.0, source="telemetry", quality="good")]
    result = estimate_feeder_state(snap, flow_measurements=fmeas)
    assert result.conflict_count >= 1
    cd = result.branches["cd"]
    assert 50.0 < cd.kw < 200.0


# ---------------------------------------------------------------------------
# Unobservable areas
# ---------------------------------------------------------------------------
def test_deenergized_buses_are_unobservable():
    snap = _feeder_with_tie()
    result = estimate_feeder_state(snap)
    d = result.buses["D"]
    assert d.observable is False
    assert d.source == "unobservable"
    assert d.confidence == 0.0


def test_unobservable_areas_reported():
    snap = _feeder_with_tie()
    result = estimate_feeder_state(snap)
    # D is de-energized (open tie ad), so it forms an unobservable area.
    all_unobs = set()
    for area in result.unobservable_areas:
        all_unobs.update(area)
    assert "D" in all_unobs


def test_open_switch_branch_unobservable():
    snap = _feeder_with_tie()
    result = estimate_feeder_state(snap)
    cd = result.branches["cd"]
    assert cd.observable is False


# ---------------------------------------------------------------------------
# Data version
# ---------------------------------------------------------------------------
def test_data_version_echoed():
    snap = _feeder()
    result = estimate_feeder_state(snap, data_version=42)
    assert result.data_version == 42


def test_data_version_takes_max_of_measurements():
    snap = _feeder()
    vmeas = [VoltageMeasurement("C", 12.45, version=7)]
    fmeas = [FlowMeasurement("cd", 50.0, 0.0, version=3)]
    result = estimate_feeder_state(snap, voltage_measurements=vmeas, flow_measurements=fmeas, data_version=5)
    assert result.data_version == 7


# ---------------------------------------------------------------------------
# Raw data unchanged
# ---------------------------------------------------------------------------
def test_raw_snapshot_not_mutated():
    snap = _feeder()
    original_switch_state = dict(snap.switch_state)
    original_buses = {b: (bus.demand_kw, bus.kv_ll) for b, bus in snap.buses.items()}
    vmeas = [VoltageMeasurement("C", 12.45)]
    fmeas = [FlowMeasurement("cd", 50.0, 0.0)]
    estimate_feeder_state(snap, voltage_measurements=vmeas, flow_measurements=fmeas)
    assert dict(snap.switch_state) == original_switch_state
    assert {b: (bus.demand_kw, bus.kv_ll) for b, bus in snap.buses.items()} == original_buses


def test_measurements_not_mutated():
    snap = _feeder()
    vmeas = [VoltageMeasurement("C", 12.45, version=1)]
    fmeas = [FlowMeasurement("cd", 50.0, 0.0, version=1)]
    estimate_feeder_state(snap, voltage_measurements=vmeas, flow_measurements=fmeas)
    assert vmeas[0].version == 1
    assert fmeas[0].version == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_missing_source_raises():
    snap = FeederSnapshot({}, {}, {})
    with pytest.raises(ValueError):
        estimate_feeder_state(snap)


def test_measurement_count():
    snap = _feeder()
    vmeas = [VoltageMeasurement("C", 12.45), VoltageMeasurement("D", 12.44)]
    fmeas = [FlowMeasurement("cd", 50.0, 0.0)]
    result = estimate_feeder_state(snap, voltage_measurements=vmeas, flow_measurements=fmeas)
    assert result.measurement_count == 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_state_estimate_returns_success(capsys):
    assert main(["state-estimate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "buses" in payload
    assert "branches" in payload
    assert "unobservable_areas" in payload
    assert payload["data_version"] >= 0
    assert set(payload["buses"]) == {"A", "B", "C", "D"}
    assert set(payload["branches"]) == {"ab", "bc", "cd"}
