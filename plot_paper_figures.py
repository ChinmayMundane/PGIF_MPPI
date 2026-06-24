import os
import json
import matplotlib.pyplot as plt
import numpy as np

import sys


def plot_paper_figures(target_dir="results"):
    results_path = f"{target_dir}/mppi_metrics.json"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run evaluate_mppi.py first.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    episodes = data["episodes"]

    os.makedirs(f"{target_dir}/plots", exist_ok=True)

    # ── Figure 1: Individual Trajectory Maps ──
    for ep in episodes:
        fig, ax = plt.subplots(figsize=(10, 4))

        # Global path
        gx = np.linspace(0, 15, 30)
        gy = np.zeros_like(gx)
        ax.plot(
            gx, gy, "--", color="gray", alpha=0.5, lw=2, label="Global Path", zorder=1
        )

        ex = ep["ego_trace_x"]
        ey = ep["ego_trace_y"]

        # Plot ego progression
        ax.plot(
            ex, ey, label="Ego Trajectory", color="#00e5ff", alpha=0.9, lw=2.5, zorder=4
        )
        ax.plot(ex[0], ey[0], "go", markersize=6, label="Start", zorder=5)

        # Plot human trajectories
        h_traces = ep["human_traces"]
        if h_traces:
            num_humans = len(h_traces[0])
            for hi in range(num_humans):
                hx = [step[hi][0] for step in h_traces]
                hy = [step[hi][1] for step in h_traces]

                lbl = "Pedestrian" if hi == 0 else ""
                ax.plot(hx, hy, color="#ff6b35", alpha=0.7, lw=1.5, label=lbl, zorder=2)
                ax.plot(hx[0], hy[0], "o", color="#ff6b35", markersize=3, zorder=3)
                ax.plot(hx[-1], hy[-1], "x", color="#ff6b35", markersize=4, zorder=3)

        # Goal and End state
        ax.plot(14.0, 0.0, "*", color="#76ff03", markersize=14, label="Goal", zorder=5)

        if ep["success"]:
            status_text = "Success"
            ax.plot(ex[-1], ey[-1], "s", color="#76ff03", markersize=6, zorder=6)
        else:
            status_text = f"Failure: {ep['failure_reason'].capitalize()}"
            ax.plot(ex[-1], ey[-1], "rX", markersize=8, label="Failure Point", zorder=6)

        ax.set_title(f"MPPI Seed {ep['seed']} — Status: {status_text}")
        ax.set_xlabel("X coordinate (m)")
        ax.set_ylabel("Y coordinate (m)")

        # Move legend outside to prevent clutter
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"{target_dir}/plots/trajectory_seed_{ep['seed']}.png", dpi=150)
        plt.close(fig)

    print(f"Saved {len(episodes)} individual trajectory plots to {target_dir}/plots/")

    # ── Summary Prints ──
    s = data["summary"]
    print("\n--- Paper Metrics Summary ---")
    print(f"Success Rate:   {s['success_rate'] * 100:.1f}%")
    print(f"Timeout Rate:   {s['timeout_rate'] * 100:.1f}%")
    print(f"Collision Rate: {s['collision_rate'] * 100:.1f}%")
    print(f"Avg Path Len:   {s['avg_path_length']:.2f} m")
    print(f"Avg Compute:    {s['avg_time_per_step_ms']:.2f} ms/step")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results"
    plot_paper_figures(target)
