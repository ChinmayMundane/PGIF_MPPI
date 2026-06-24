import os
import json
import numpy as np
from mppi_dynamic_humans import run_simulation


def evaluate_seeds(
    num_seeds=5,
    max_steps=200,
    save_dir="results",
    use_gaussian_cost=True,
    difficulty="hard",
):
    """
    Run the simulation across multiple seeds to gather benchmark data.
    """
    results = []

    # Enable JAX fast execution silently
    os.environ["JAX_PLATFORMS"] = "cpu"

    print("\n========================================================")
    print(f"Starting headless evaluation across {num_seeds} seeds")
    print(
        f"Save dir: {save_dir} | Gaussian cost: {use_gaussian_cost} | Diff: {difficulty.upper()}"
    )
    print("========================================================\n")

    for seed in range(num_seeds):
        print(f"\n--- Running Seed {seed} ---")

        # Run the head-less simulation
        frames, global_path, humans, x_goal, y_goal, collision, elapsed = (
            run_simulation(
                seed=seed,
                headless=True,
                max_steps=max_steps,
                randomize=True,
                use_gaussian_cost=use_gaussian_cost,
                difficulty=difficulty,
            )
        )

        # Calculate metrics
        num_steps = len(frames)
        total_time = elapsed
        time_per_step = total_time / num_steps if num_steps > 0 else 0

        # Extract ego trajectory
        ego_x = [f["ego_x"] for f in frames]
        ego_y = [f["ego_y"] for f in frames]

        # Reconstruct path length
        path_length = 0.0
        for i in range(1, num_steps):
            path_length += np.hypot(ego_x[i] - ego_x[i - 1], ego_y[i] - ego_y[i - 1])

        success = (
            np.hypot(ego_x[-1] - x_goal, ego_y[-1] - y_goal) <= 0.5
        ) and not collision

        # Calculate safety metrics (minimum distance to any human throughout the run)
        min_human_distance = float("inf")
        for f in frames:
            ex, ey = f["ego_x"], f["ego_y"]
            for hx, hy, hvx, hvy in f["humans"]:
                dist = np.hypot(ex - hx, ey - hy)
                if dist < min_human_distance:
                    min_human_distance = float(dist)

        failure_reason = "none"
        if not success:
            failure_reason = "collision" if collision else "timeout"

        # Log episode metrics
        ep_result = {
            "seed": seed,
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(failure_reason == "timeout"),
            "failure_reason": failure_reason,
            "steps": num_steps,
            "path_length": float(path_length),
            "time_per_step_ms": float(time_per_step * 1000.0),
            "min_human_distance": float(min_human_distance),
            "ego_trace_x": ego_x,
            "ego_trace_y": ego_y,
            "human_traces": [
                [[float(h[0]), float(h[1])] for h in f["humans"]] for f in frames
            ],
        }

        print(
            f"Seed {seed} finished - {failure_reason.upper()} - Length: {path_length:.2f}m"
        )
        results.append(ep_result)

    # Aggregate output
    output_data = {
        "summary": {
            "num_seeds": num_seeds,
            "success_rate": float(np.mean([r["success"] for r in results])),
            "collision_rate": float(np.mean([r["collision"] for r in results])),
            "timeout_rate": float(np.mean([r["timeout"] for r in results])),
            "avg_path_length": float(np.mean([r["path_length"] for r in results])),
            "avg_time_per_step_ms": float(
                np.mean([r["time_per_step_ms"] for r in results])
            ),
        },
        "episodes": results,
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/mppi_metrics.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nBenchmark complete! Results saved to {save_dir}/mppi_metrics.json")
    print(json.dumps(output_data["summary"], indent=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_steps", type=int, default=200, help="Maximum number of simulation steps"
    )
    args = parser.parse_args()

    evaluate_seeds(
        num_seeds=100,
        max_steps=args.max_steps,
        save_dir="results_easy",
        use_gaussian_cost=True,
        difficulty="easy",
    )
    evaluate_seeds(
        num_seeds=100,
        max_steps=args.max_steps,
        save_dir="results_medium",
        use_gaussian_cost=True,
        difficulty="medium",
    )
    evaluate_seeds(
        num_seeds=100,
        max_steps=args.max_steps,
        save_dir="results_hard",
        use_gaussian_cost=True,
        difficulty="hard",
    )
