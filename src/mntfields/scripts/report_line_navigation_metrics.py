#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_MODEL_DIR = Path("/home/mayank/turtlebot3_ws/mntfields_model")
DEFAULT_LINE_METRICS = DEFAULT_MODEL_DIR / "line_navigation_metrics.json"
DEFAULT_SDF_METRICS = DEFAULT_MODEL_DIR / "sdf_replanner_metrics.json"


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt_stat_block(name: str, stats: dict | None, unit: str) -> str:
    if not stats or int(stats.get("count", 0)) == 0:
        return f"  {name}: no data"
    return (
        f"  {name}: count={stats['count']} "
        f"mean={stats['mean']:.3f} {unit} "
        f"median={stats['median']:.3f} {unit} "
        f"min={stats['min']:.3f} {unit} "
        f"max={stats['max']:.3f} {unit}"
    )


def main():
    parser = argparse.ArgumentParser(description="Report line navigation + SDF metrics")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=None,
        help="Optional model directory containing line_navigation_metrics.json and sdf_replanner_metrics.json",
    )
    parser.add_argument("--line-metrics", default=str(DEFAULT_LINE_METRICS))
    parser.add_argument("--sdf-metrics", default=str(DEFAULT_SDF_METRICS))
    parser.add_argument("--segments", action="store_true", help="Show per-plan segment timings")
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        line_path = model_dir / "line_navigation_metrics.json"
        sdf_path = model_dir / "sdf_replanner_metrics.json"
    else:
        line_path = Path(args.line_metrics).expanduser().resolve()
        sdf_path = Path(args.sdf_metrics).expanduser().resolve()
    line = load_json(line_path)
    sdf = load_json(sdf_path)

    if line is None and sdf is None:
        raise SystemExit("No metrics files found.")

    if line is not None:
        print("Line Navigation")
        print(f"  file: {line_path}")
        print(f"  run_duration_s: {float(line.get('run_duration_s', 0.0)):.3f}")
        summary = line.get("summary", {})
        print("Timing")
        print(fmt_stat_block("trajectory_generation_ms", summary.get("trajectory_generation_ms"), "ms"))
        per_region = summary.get("trajectory_generation_ms_by_region", {})
        if per_region:
            print("Per Region")
            for ver in sorted(per_region, key=lambda x: int(x)):
                print(fmt_stat_block(f"R{ver}", per_region[ver], "ms"))
        if args.segments:
            print("Plans")
            for idx, plan in enumerate(line.get("trajectory_plan_metrics", []), start=1):
                status = "ok" if plan.get("success") else f"fail:{plan.get('failure_reason')}"
                print(
                    f"  {idx}. goal_request_id={plan.get('goal_request_id')} "
                    f"planner={plan.get('planner_mode')} total={plan.get('total_plan_ms')}ms "
                    f"path_pts={plan.get('path_points')} status={status}"
                )
                for seg in plan.get("segments", []):
                    print(
                        f"     R{seg.get('region_version')}: "
                        f"{seg.get('field_generate_ms')}ms "
                        f"pts={seg.get('segment_points')} "
                        f"fallback={seg.get('fallback_used')}"
                    )
        print()

    if sdf is not None:
        print("SDF Replanner")
        print(f"  file: {sdf_path}")
        print(f"  run_duration_s: {float(sdf.get('run_duration_s', 0.0)):.3f}")
        summary = sdf.get("summary", {})
        print("Timing")
        print(fmt_stat_block("sdf_optimizer_inference_ms", summary.get("sdf_optimizer_inference_ms"), "ms"))
        print(fmt_stat_block("sdf_total_cycle_ms", summary.get("sdf_total_cycle_ms"), "ms"))
        if args.segments:
            print("Cycles")
            for idx, cycle in enumerate(sdf.get("sdf_optimization_metrics", []), start=1):
                status = "published" if cycle.get("published") else f"drop:{cycle.get('failure_reason')}"
                print(
                    f"  {idx}. infer={cycle.get('optimizer_inference_ms')}ms "
                    f"total={cycle.get('total_sdf_cycle_ms')}ms "
                    f"ahead_pts={cycle.get('ahead_segment_points')} status={status}"
                )


if __name__ == "__main__":
    main()
