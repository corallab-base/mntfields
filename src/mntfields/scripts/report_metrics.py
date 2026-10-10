#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def fmt_num(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def fmt_stats(name, stats, unit=""):
    if not stats or stats.get("count", 0) == 0:
        return f"{name}: no data"
    suffix = f" {unit}".rstrip()
    return (
        f"{name}: count={stats['count']} "
        f"mean={fmt_num(stats.get('mean'))}{suffix} "
        f"median={fmt_num(stats.get('median'))}{suffix} "
        f"min={fmt_num(stats.get('min'))}{suffix} "
        f"max={fmt_num(stats.get('max'))}{suffix}"
    )


def print_summary(data):
    summary = data.get("summary", {})
    print("Run")
    print(f"  duration_s: {fmt_num(data.get('run_duration_s'))}")
    print(f"  lidar_drop_count: {fmt_num(data.get('lidar_drop_count'))}")
    print(f"  lidar_drop_max_lag_s: {fmt_num(data.get('lidar_drop_max_lag_s'))}")
    print()

    print("Goals")
    print(f"  total: {fmt_num(summary.get('goals_total'))}")
    print(f"  reached: {fmt_num(summary.get('goals_reached'))}")
    print(f"  goal_success_rate: {fmt_num(summary.get('goal_success_rate'))}")
    print(f"  motion_success_rate: {fmt_num(summary.get('motion_success_rate'))}")
    print()

    print("Timing")
    print(f"  {fmt_stats('frontier_pick_ms', summary.get('frontier_pick_ms'), 'ms')}")
    print(f"  {fmt_stats('path_gen_ms', summary.get('path_gen_ms'), 'ms')}")
    print(f"  {fmt_stats('train_tick_ms', summary.get('train_tick_ms'), 'ms')}")
    print(f"  {fmt_stats('time_to_first_path_s', summary.get('time_to_first_path_s'), 's')}")
    print(f"  {fmt_stats('time_to_first_motion_s', summary.get('time_to_first_motion_s'), 's')}")
    print(f"  {fmt_stats('time_to_goal_s', summary.get('time_to_goal_s'), 's')}")
    print(f"  {fmt_stats('time_in_rotate_only_s', summary.get('time_in_rotate_only_s'), 's')}")
    print(f"  {fmt_stats('replans_per_goal', summary.get('replans_per_goal'))}")
    print()

    print("Regions")
    for ver, region in sorted(data.get("regions", {}).items(), key=lambda kv: int(kv[0])):
        ticks = region.get("train_tick_ms", [])
        print(
            f"  R{ver}: epoch={fmt_num(region.get('epoch'))} "
            f"train_ticks={len(ticks)} loss_history_len={fmt_num(region.get('loss_history_len'))}"
        )


def print_goals(data):
    goals = data.get("goal_metrics", [])
    if not goals:
        print("No goal metrics.")
        return
    print("Goal Details")
    for i, goal in enumerate(goals, start=1):
        xy = goal.get("goal_xy", [None, None])
        print(
            f"  {i}. goal=({fmt_num(xy[0], 2)},{fmt_num(xy[1], 2)}) "
            f"result={goal.get('result', '-')}"
            f" start_dist={fmt_num(goal.get('distance_at_start_m'))}m"
            f" finish_dist={fmt_num(goal.get('distance_at_finish_m'))}m"
            f" first_path={fmt_num(goal.get('time_to_first_path_s'))}s"
            f" first_motion={fmt_num(goal.get('time_to_first_motion_s'))}s"
            f" total={fmt_num(goal.get('time_to_goal_s'))}s"
            f" rotate_only={fmt_num(goal.get('time_in_rotate_only_s'))}s"
            f" replans={len(goal.get('replan_labels', []))}"
        )


def main():
    parser = argparse.ArgumentParser(description="Report mntfields metrics.json")
    parser.add_argument(
        "--metrics",
        default="/home/mayank/turtlebot3_ws/mntfields_model/metrics.json",
        help="Path to metrics.json",
    )
    parser.add_argument(
        "--goals",
        action="store_true",
        help="Print goal-level details too",
    )
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")

    with path.open() as f:
        data = json.load(f)

    print_summary(data)
    if args.goals:
        print()
        print_goals(data)


if __name__ == "__main__":
    main()
