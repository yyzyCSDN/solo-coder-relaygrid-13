import copy

from relaygrid.models import Branch, Bus, FeederSnapshot, SwitchState
from relaygrid.stateest.estimator import (
    BranchFlowEstimate,
    Conflict,
    FlowDatum,
    StateEstimate,
    SwitchDatum,
    VoltageDatum,
    estimate_feeder_state,
)


def four_bus_feeder():
    buses = {
        "A": Bus("A", 12.47, customers=1, demand_kw=0.0, demand_kvar=0.0),
        "B": Bus("B", 12.47, customers=50, demand_kw=200.0, demand_kvar=60.0),
        "C": Bus("C", 12.47, customers=40, demand_kw=150.0, demand_kvar=45.0),
        "D": Bus("D", 12.47, customers=30, demand_kw=100.0, demand_kvar=30.0),
    }
    branches = {
        "ab": Branch("ab", "A", "B", 0.15, 0.10, 400.0),
        "bc": Branch("bc", "B", "C", 0.20, 0.12, 400.0),
        "cd": Branch("cd", "C", "D", 0.25, 0.15, 300.0, True, False),
        "ad": Branch("ad", "A", "D", 0.30, 0.18, 300.0, True, False),
    }
    return FeederSnapshot(
        buses, branches,
        {"cd": SwitchState.OPEN, "ad": SwitchState.OPEN},
    )


def test_scada_voltage_and_flow_anchor_bus_estimates():
    snap = four_bus_feeder()
    est = estimate_feeder_state(
        snap,
        voltages=[VoltageDatum("v_a", "A", 12.47, version=3)],
        flows=[FlowDatum("f_ab", "ab", kw=450.0, kvar=135.0, version=2)],
        now_s=0.0,
    )
    assert isinstance(est, StateEstimate)
    assert est.bus_voltages["A"].observable
    assert abs(est.bus_voltages["A"].kv_ll - 12.47) < 1e-9
    # Whole energized component A-B-C is observable from the A anchor.
    for bus in "ABC":
        v = est.bus_voltages[bus]
        assert v.observable and v.kv_ll is not None and 0.9 < v.pu < 1.1
    # Downstream voltages must not exceed the source under positive flow.
    assert est.bus_voltages["B"].kv_ll < est.bus_voltages["A"].kv_ll
    assert est.bus_voltages["C"].kv_ll < est.bus_voltages["B"].kv_ll
    # D is isolated by the two open tie switches -> its own component.
    assert not est.bus_voltages["D"].observable
    assert est.bus_voltages["D"].kv_ll is None
    region_d = next(r for r in est.unobservable_regions if r.buses == ("D",))
    # The region report names the open boundary switches that isolate it.
    assert set(region_d.branches) == {"ad", "cd"}

    ab = est.branch_flows["ab"]
    assert ab.observable and ab.kw is not None and ab.confidence > 0.5
    # bc and pseudo-only branches stay observable with low confidence.
    assert est.branch_flows["bc"].observable
    assert est.branch_flows["bc"].confidence < est.branch_flows["ab"].confidence
    # Open ties: zero flow asserted from topology.
    for tie in ("cd", "ad"):
        f = est.branch_flows[tie]
        assert f.observable and f.kw == 0.0 and f.current_a == 0.0


def test_unobservable_region_reported_without_anchor():
    snap = four_bus_feeder()
    est = estimate_feeder_state(snap, now_s=0.0)
    regions = est.unobservable_regions
    region_buses = {tuple(r.buses) for r in regions}
    assert ("D",) in region_buses
    assert tuple(sorted("ABC")) in region_buses
    assert all(r.reason == "no_voltage_anchor" for r in regions)
    # Branches inside an unobservable component get no flow estimate.
    assert est.branch_flows["ab"].kw is None
    assert est.branch_flows["ab"].observable is False


def test_ami_voltage_lower_weight_than_scada():
    snap = four_bus_feeder()
    est = estimate_feeder_state(
        snap,
        voltages=[
            VoltageDatum("v_scada", "B", 12.40, source="scada",
                         quality="good", version=1),
            VoltageDatum("v_ami", "B", 12.00, source="ami",
                         quality="good", version=1),
        ],
        now_s=0.0,
    )
    # SCADA weight 0.85 vs AMI 0.55: estimate is pulled toward SCADA.
    v = est.bus_voltages["B"].kv_ll
    assert 12.3 < v < 12.4
    assert any(c.kind == "voltage" and c.element_id == "B"
               for c in est.conflicts)


def test_conflict_downweights_lower_quality_source():
    snap = four_bus_feeder()
    est = estimate_feeder_state(
        snap,
        voltages=[
            VoltageDatum("v_a_good", "A", 12.47, quality="good", version=2),
            VoltageDatum("v_a_bad", "A", 11.0, quality="suspect", version=1),
        ],
        now_s=0.0,
    )
    # Suspect 11.0 kV (huge deviation) loses the conflict and is down-weighted.
    assert abs(est.bus_voltages["A"].kv_ll - 12.47) < 0.1
    conflict = next(c for c in est.conflicts
                    if c.kind == "voltage" and c.element_id == "A")
    assert "v_a_bad" in conflict.point_ids
    # Both versions are still reported as used data versions.
    assert est.versions["v_a_good"] == 2
    assert est.versions["v_a_bad"] == 1


def test_stale_measurements_lose_weight():
    snap = four_bus_feeder()
    est = estimate_feeder_state(
        snap,
        voltages=[
            VoltageDatum("v_fresh", "A", 12.47, timestamp_s=100, version=2),
            VoltageDatum("v_old", "A", 12.0, timestamp_s=0, version=1),
        ],
        now_s=110.0, max_age_s=120.0,
    )
    # Old reading is only 10 s from expiring; fresh one dominates.
    assert abs(est.bus_voltages["A"].kv_ll - 12.47) < 0.05


def test_switch_position_contradiction_adopts_high_quality():
    snap = four_bus_feeder()
    # A telemetered CLOSE on tie "ad" energizes D from A; good SCADA wins
    # over a stale suspect OPEN indication.
    est = estimate_feeder_state(
        snap,
        voltages=[VoltageDatum("v_a", "A", 12.47, version=1)],
        switches=[
            SwitchDatum("sw_ad_close", "ad", SwitchState.CLOSED,
                        quality="good", version=5, timestamp_s=10.0),
            SwitchDatum("sw_ad_open", "ad", SwitchState.OPEN,
                        quality="suspect", version=4, timestamp_s=0.0),
        ],
        now_s=10.0,
    )
    assert est.bus_voltages["D"].observable
    ad = est.branch_flows["ad"]
    assert ad.observable and ad.kw is not None
    assert any(c.kind == "switch" and c.element_id == "ad"
               for c in est.conflicts)
    assert est.versions["sw_ad_close"] == 5


def test_open_switch_creates_unobservable_component():
    snap = four_bus_feeder()
    # Open the non-tie section ab: B-C island has no anchor.
    openable = FeederSnapshot(
        dict(snap.buses), dict(snap.branches),
        {**snap.switch_state, "ab": SwitchState.OPEN},
    )
    # branch ab is not switchable in the model; rebuild it switchable.
    br = snap.branches["ab"]
    openable.branches["ab"] = Branch(br.branch_id, br.source, br.target,
                                     br.r_ohm, br.x_ohm, br.ampacity_a, True)
    est = estimate_feeder_state(
        openable,
        voltages=[VoltageDatum("v_a", "A", 12.47)],
        switches=[SwitchDatum("sw_ab", "ab", SwitchState.OPEN, version=2)],
        now_s=0.0,
    )
    assert est.bus_voltages["A"].observable
    assert not est.bus_voltages["B"].observable
    assert not est.bus_voltages["C"].observable
    assert est.branch_flows["ab"].kw == 0.0


def test_raw_inputs_remain_unchanged():
    snap = four_bus_feeder()
    snap_before = copy.deepcopy(snap)
    voltages = [VoltageDatum("v_a", "A", 12.47)]
    flows = [FlowDatum("f_ab", "ab", kw=450.0, kvar=135.0)]
    switches = [SwitchDatum("sw_ad", "ad", SwitchState.OPEN)]
    estimate_feeder_state(snap, voltages, flows, switches, now_s=0.0)
    assert snap == snap_before
    assert voltages[0] == VoltageDatum("v_a", "A", 12.47)
    assert switches[0].state == SwitchState.OPEN


def test_versions_and_as_of_reported():
    snap = four_bus_feeder()
    est = estimate_feeder_state(
        snap,
        voltages=[
            VoltageDatum("v_a", "A", 12.47, timestamp_s=12.0, version=7),
            VoltageDatum("v_ami_c", "C", 12.1, source="ami",
                         timestamp_s=9.0, version=3),
        ],
        flows=[FlowDatum("f_ab", "ab", kw=450.0, kvar=135.0, version=4)],
    )
    assert est.as_of_s == 12.0
    assert est.versions["v_a"] == 7
    assert est.versions["v_ami_c"] == 3
    assert est.versions["f_ab"] == 4
    assert 0.0 <= est.confidence <= 1.0


def test_confidence_higher_with_redundant_good_sources():
    snap = four_bus_feeder()
    one = estimate_feeder_state(
        snap, voltages=[VoltageDatum("v_a", "A", 12.47)], now_s=0.0)
    two = estimate_feeder_state(
        snap,
        voltages=[
            VoltageDatum("v_a1", "A", 12.47, version=1),
            VoltageDatum("v_a2", "A", 12.46, version=2),
        ],
        now_s=0.0,
    )
    assert two.bus_voltages["A"].confidence > one.bus_voltages["A"].confidence


def test_unknown_references_rejected():
    snap = four_bus_feeder()
    import pytest
    with pytest.raises(ValueError):
        estimate_feeder_state(
            snap, voltages=[VoltageDatum("v", "ZZ", 12.47)], now_s=0.0)
    with pytest.raises(ValueError):
        estimate_feeder_state(
            snap, flows=[FlowDatum("f", "zz", kw=1.0)], now_s=0.0)
    with pytest.raises(ValueError):
        estimate_feeder_state(
            snap, switches=[SwitchDatum("s", "zz", SwitchState.OPEN)],
            now_s=0.0)


def test_as_dict_is_json_serialisable():
    import json
    snap = four_bus_feeder()
    est = estimate_feeder_state(
        snap,
        voltages=[
            VoltageDatum("v_a", "A", 12.47, version=1),
            VoltageDatum("v_ami_d", "D", 12.2, source="ami", version=1),
        ],
        flows=[FlowDatum("f_ab", "ab", kw=450.0, kvar=135.0,
                         current_a=21.5, version=1)],
        now_s=0.0,
    )
    payload = est.as_dict()
    json.dumps(payload)
    assert payload["bus_voltages"]["A"]["observable"] is True
    assert payload["bus_voltages"]["D"]["observable"] is True
    assert payload["versions"]["v_ami_d"] == 1
    assert isinstance(payload["unobservable_regions"], list)


def test_current_only_datum_uses_pseudo_power_factor():
    snap = four_bus_feeder()
    est = estimate_feeder_state(
        snap,
        voltages=[VoltageDatum("v_a", "A", 12.47)],
        flows=[FlowDatum("f_ab_i", "ab", current_a=21.5, version=1)],
        now_s=0.0,
    )
    ab = est.branch_flows["ab"]
    # sqrt(3)*12.47 kV*21.5 A ~= 463 kVA split on 200/60 pseudo PF.
    assert ab.kw > 400.0
    assert ab.kvar > 100.0
