from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from relaygrid.models import Branch, Bus, FeederSnapshot, SwitchState
from relaygrid.sim.engine import GridSimulator
from relaygrid.state_estimation import (
    FlowMeasurement,
    VoltageMeasurement,
    estimate_feeder_state,
)


def demo_summary() -> dict[str, object]:
    """Run a deterministic switching example and return its public summary."""
    buses = {bus_id: Bus(bus_id, 12.47) for bus_id in "ABCD"}
    branches = {
        "ab": Branch("ab", "A", "B", 0.1, 0.1, 300.0),
        "bc": Branch("bc", "B", "C", 0.1, 0.1, 300.0),
        "cd": Branch("cd", "C", "D", 0.1, 0.1, 300.0, True, False),
        "ad": Branch("ad", "A", "D", 0.1, 0.1, 300.0, True, False),
    }
    snapshot = FeederSnapshot(
        buses,
        branches,
        {"cd": SwitchState.OPEN, "ad": SwitchState.OPEN},
    )
    simulator = GridSimulator(snapshot)
    energized_before = simulator.energized_bus_count()
    simulator.operate_switch("cd", SwitchState.CLOSED)
    return {
        "energized_after": simulator.energized_bus_count(),
        "energized_before": energized_before,
        "events": len(simulator.log.events),
        "project": "RelayGrid",
        "switch": "cd",
        "switch_state": snapshot.switch_state["cd"].value,
    }


def state_estimate_summary() -> dict[str, object]:
    """Run the feeder state estimator on a demo feeder and return a summary."""
    buses = {
        "A": Bus("A", 12.47),
        "B": Bus("B", 12.47),
        "C": Bus("C", 12.47, demand_kw=100.0),
        "D": Bus("D", 12.47, demand_kw=50.0),
    }
    branches = {
        "ab": Branch("ab", "A", "B", 0.1, 0.1, 300.0),
        "bc": Branch("bc", "B", "C", 0.1, 0.1, 300.0),
        "cd": Branch("cd", "C", "D", 0.1, 0.1, 300.0, True, False),
    }
    snapshot = FeederSnapshot(buses, branches, {})
    vmeas = [VoltageMeasurement("C", 12.45, source="telemetry", quality="good")]
    fmeas = [FlowMeasurement("cd", 50.0, 0.0, source="telemetry", quality="good")]
    result = estimate_feeder_state(
        snapshot,
        voltage_measurements=vmeas,
        flow_measurements=fmeas,
        data_version=1,
    )
    return {
        "buses": {
            bid: {
                "kv": round(est.kv, 4),
                "confidence": round(est.confidence, 3),
                "observable": est.observable,
                "source": est.source,
            }
            for bid, est in sorted(result.buses.items())
        },
        "branches": {
            bid: {
                "kw": round(est.kw, 2),
                "kvar": round(est.kvar, 2),
                "confidence": round(est.confidence, 3),
                "observable": est.observable,
            }
            for bid, est in sorted(result.branches.items())
        },
        "unobservable_areas": [sorted(area) for area in result.unobservable_areas],
        "data_version": result.data_version,
        "measurement_count": result.measurement_count,
        "conflict_count": result.conflict_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relaygrid")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("demo", help="run a deterministic feeder demo")
    subcommands.add_parser(
        "state-estimate", help="run feeder state estimation on a demo feeder"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        print(json.dumps(demo_summary(), sort_keys=True))
        return 0
    if args.command == "state-estimate":
        print(json.dumps(state_estimate_summary(), sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
