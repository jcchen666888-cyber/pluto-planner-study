"""Create a compact, reviewable JSON summary from a one-scenario nuPlan run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aggregate_files = list((args.run_dir / "aggregator_metric").glob("*.parquet"))
    if len(aggregate_files) != 1:
        raise RuntimeError(f"expected one aggregate parquet, found {aggregate_files}")
    aggregate = pd.read_parquet(aggregate_files[0])
    final = aggregate.loc[aggregate["scenario"] == "final_score"].iloc[0]
    runner = pd.read_parquet(args.run_dir / "runner_report.parquet").iloc[0]

    report = {
        "status": "pass" if bool(runner["succeeded"]) else "fail",
        "scope": "one nuPlan mini closed-loop non-reactive scenario; not a benchmark reproduction",
        "planner": str(final["planner_name"]),
        "scenario_token": str(runner["scenario_name"]),
        "scenario_type": str(aggregate.iloc[0]["scenario_type"]),
        "log_name": str(runner["log_name"]),
        "score": float(final["score"]),
        "metrics": {
            "ego_progress_along_expert_route": float(final["ego_progress_along_expert_route"]),
            "no_ego_at_fault_collisions": float(final["no_ego_at_fault_collisions"]),
            "drivable_area_compliance": float(final["drivable_area_compliance"]),
            "driving_direction_compliance": float(final["driving_direction_compliance"]),
            "ego_is_comfortable": float(final["ego_is_comfortable"]),
            "ego_is_making_progress": float(final["ego_is_making_progress"]),
            "speed_limit_compliance": float(final["speed_limit_compliance"]),
            "time_to_collision_within_bound": float(final["time_to_collision_within_bound"]),
        },
        "runtime_seconds": {
            "scenario": float(runner["duration"]),
            "planning_step_mean": float(runner["inference_runtimes_mean"]),
            "planning_step_median": float(runner["inference_runtimes_median"]),
            "planning_step_std": float(runner["inference_runtimes_std"]),
        },
        "source": aggregate_files[0].name,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
