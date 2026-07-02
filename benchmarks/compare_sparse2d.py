"""
compare_sparse2d.py

Comprehensive benchmark sweep comparing WaveHoltz and SparseDirect subdomain
solvers across mesh sizes and kernel configurations, with two studies:

  kernel — Per-action timing (DDSubstructuredOperator::action()).
           Sweeps mesh sizes × selected kernel configs.
           Generates throughput and runtime scaling plots, plus SparseDirect
           factorization cost and memory plots.

  e2e    — Full end-to-end DDH solve (build + outer MINRES).
           Sweeps mesh sizes with a fixed subdomain element dimension.
           Generates build/solve breakdown, iteration count, memory, and
           speedup plots.

Usage
─────
  python benchmarks/compare_sparse2d.py [options]

  --exe PATH          Path to compare_sparse2d executable
                      (default: build/benchmarks/compare_sparse2d)
  --output-dir PATH   Directory for combined CSVs and plots
                      (default: benchmarks/results/compare_sparse2d)
  --degree INT        Polynomial degree (default: 3)
  --warmup INT        [kernel] Warmup iterations (default: 5)
  --iterations INT    [kernel] Timed iterations per run (default: 20)
  --maxit INT         [e2e] Max outer MINRES iterations (default: 1000)
  --rtol FLOAT        [e2e] Relative tolerance for outer MINRES (default: 1e-3)
  --precision STR     Scalar precision: float or double (default: float)
  --mode STR          Study to run: kernel | e2e | all (default: all)
  --skip-runs         Skip execution; regenerate plots from existing CSVs
  --max-kernel-mesh   Max mesh side for kernel study (default: 256)
  --max-e2e-mesh      Max mesh side for E2E study (default: 128)
  --e2e-sub INT       Subdomain element side for E2E study (default: 8)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mesh sweep definitions
# ---------------------------------------------------------------------------

ALL_MESHES: list[tuple[int, int]] = [
    (16, 16),
    (32, 32),
    (48, 48),
    (64, 64),
    (96, 96),
    (128, 128),
    (192, 192),
    (256, 256),
    (384, 384),
    (512, 512),
]

KERNEL_BLOCK_SIZES: list[int] = [256, 512, 1024]
KERNEL_TDOFS: list[int] = [1, 2, 4]
ALL_KERNEL_CONFIGS: list[tuple[int, int]] = [(b, k) for b in KERNEL_BLOCK_SIZES for k in KERNEL_TDOFS]
SELECTED_KERNEL_CONFIGS: list[tuple[int, int]] = [(256, 1), (1024, 1), (1024, 4)]


# ---------------------------------------------------------------------------
# Subdomain sizing helpers (mirrors compare_minres2d.py)
# ---------------------------------------------------------------------------


def kernel_capacity_elements(degree: int, block_size: int, tdof: int) -> int:
    """Number of subdomain elements a single WaveHoltz kernel block can handle."""
    p2 = (degree + 1) ** 2
    return (block_size // p2) * tdof


def choose_subdomain_dims(nx: int, ny: int, target_elems: int) -> tuple[int, int]:
    """Largest sx×sy with sx·sy ≤ target_elems, matching the mesh aspect ratio."""
    if target_elems <= 1:
        return 1, 1
    max_elems = min(target_elems, nx * ny)
    target_aspect = nx / ny
    best_sx, best_sy = 1, 1
    best_fill = 1
    best_aspect_err = float("inf")
    for sx in range(1, min(nx, max_elems) + 1):
        sy = min(ny, max_elems // sx)
        if sy < 1:
            continue
        fill = sx * sy
        if fill < best_fill:
            continue
        aspect_err = abs(math.log(sx / sy / target_aspect))
        if fill > best_fill or aspect_err < best_aspect_err:
            best_sx, best_sy = sx, sy
            best_fill = fill
            best_aspect_err = aspect_err
    return best_sx, best_sy


def subdomains_for_kernel(nx: int, ny: int, degree: int, block_size: int, tdof: int) -> tuple[int, int] | None:
    cap = kernel_capacity_elements(degree, block_size, tdof)
    if cap <= 0:
        return None
    return choose_subdomain_dims(nx, ny, cap)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------


def configure_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.8),
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "axes.grid": True,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "legend.framealpha": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _set_dof_xaxis(ax: matplotlib.axes.Axes) -> None:
    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def _style_axes(ax: matplotlib.axes.Axes) -> None:
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)


def _kernel_title(block_size: int, tdof: int) -> str:
    return f"$B = {block_size},\\, T = {tdof}$"


def _kernel_slug(block_size: int, tdof: int) -> str:
    return f"B{block_size}_T{tdof}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep compare_sparse2d and plot WaveHoltz vs SparseDirect scaling."
    )
    parser.add_argument(
        "--exe",
        default=str(Path("build") / "benchmarks" / "compare_sparse2d"),
        help="Path to compare_sparse2d executable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("benchmarks") / "results" / "compare_sparse2d"),
        help="Directory where CSVs and plots are written.",
    )
    parser.add_argument("--degree", type=int, default=3, help="Polynomial degree.")
    parser.add_argument("--warmup", type=int, default=5, help="[kernel] Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=20, help="[kernel] Timed iterations per run.")
    parser.add_argument("--maxit", type=int, default=1000, help="[e2e] Max outer MINRES iterations.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="[e2e] Relative tolerance for outer MINRES.")
    parser.add_argument(
        "--precision",
        choices=["float", "double"],
        default="float",
        help="Scalar precision.",
    )
    parser.add_argument(
        "--mode",
        choices=["kernel", "e2e", "all"],
        default="all",
        help="Study to run: kernel | e2e | all.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip execution; regenerate plots from existing combined CSVs.",
    )
    parser.add_argument(
        "--max-kernel-mesh",
        type=int,
        default=256,
        help="Max mesh side for kernel study.",
    )
    parser.add_argument(
        "--max-e2e-mesh",
        type=int,
        default=128,
        help="Max mesh side for E2E study.",
    )
    parser.add_argument(
        "--e2e-sub",
        type=int,
        default=8,
        help="Subdomain element side (square) for E2E study.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Kernel study: plan + run
# ---------------------------------------------------------------------------


def build_kernel_plan(degree: int, precision: str, max_mesh: int) -> list[dict]:
    plan: list[dict] = []
    run_id = 1
    meshes = [(nx, ny) for (nx, ny) in ALL_MESHES if nx <= max_mesh]
    for nx, ny in meshes:
        for block_size, tdof in ALL_KERNEL_CONFIGS:
            sub = subdomains_for_kernel(nx, ny, degree, block_size, tdof)
            if sub is None:
                continue
            sx, sy = sub
            plan.append(
                {
                    "run_id": run_id,
                    "precision": precision,
                    "degree": degree,
                    "nx": nx,
                    "ny": ny,
                    "sx": sx,
                    "sy": sy,
                    "block_size": block_size,
                    "tdof": tdof,
                }
            )
            run_id += 1
    return plan


def run_kernel_benchmark(
    exe: Path,
    run: dict,
    warmup: int,
    iterations: int,
    row_file: Path,
) -> dict:
    cmd = [
        str(exe),
        "--mode", "kernel",
        "--precision", run["precision"],
        "--degree", str(run["degree"]),
        "--mesh", str(run["nx"]), str(run["ny"]),
        "--subdomains", str(run["sx"]), str(run["sy"]),
        "--block-size", str(run["block_size"]),
        "--tdof", str(run["tdof"]),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--output", str(row_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"run_id={run['run_id']} failed\ncmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    row_df = pd.read_csv(row_file)
    if row_df.shape[0] != 1:
        raise RuntimeError(f"expected 1 row in {row_file}, got {row_df.shape[0]}")
    row = row_df.iloc[0].to_dict()
    row["run_id"] = run["run_id"]
    row["requested_block_size"] = run["block_size"]
    row["requested_tdof"] = run["tdof"]
    return row


def run_kernel_study(
    exe: Path,
    output_dir: Path,
    degree: int,
    precision: str,
    warmup: int,
    iterations: int,
    max_mesh: int,
) -> pd.DataFrame:
    rows_dir = output_dir / "kernel_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    plan = build_kernel_plan(degree=degree, precision=precision, max_mesh=max_mesh)
    print(f"[kernel] Planned runs: {len(plan)}")

    rows, failures = [], []
    t0 = time.time()
    for idx, run in enumerate(plan, start=1):
        row_file = rows_dir / f"run_{run['run_id']:04d}.csv"
        if row_file.exists():
            row_file.unlink()
        print(
            f"[{idx:03d}/{len(plan):03d}] mesh={run['nx']}x{run['ny']}, "
            f"sub={run['sx']}x{run['sy']}, B={run['block_size']}, T={run['tdof']}"
        )
        try:
            rows.append(run_kernel_benchmark(exe=exe, run=run, warmup=warmup, iterations=iterations, row_file=row_file))
        except RuntimeError as err:
            failures.append({"run_id": run["run_id"], "error": str(err)})
            print(f"  -> FAILED: {err}")

    elapsed = time.time() - t0
    print(f"[kernel] {len(rows)} runs in {elapsed / 60:.1f} min; {len(failures)} failures")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "kernel_failures.csv", index=False)
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# E2E study: plan + run
# ---------------------------------------------------------------------------


def build_e2e_plan(degree: int, precision: str, max_mesh: int, sub_side: int, block_size: int, tdof: int) -> list[dict]:
    plan: list[dict] = []
    run_id = 1
    meshes = [(nx, ny) for (nx, ny) in ALL_MESHES if nx <= max_mesh]
    for nx, ny in meshes:
        plan.append(
            {
                "run_id": run_id,
                "precision": precision,
                "degree": degree,
                "nx": nx,
                "ny": ny,
                "sx": sub_side,
                "sy": sub_side,
                "block_size": block_size,
                "tdof": tdof,
            }
        )
        run_id += 1
    return plan


def run_e2e_benchmark(
    exe: Path,
    run: dict,
    maxit: int,
    rtol: float,
    row_file: Path,
) -> dict:
    cmd = [
        str(exe),
        "--mode", "e2e",
        "--precision", run["precision"],
        "--degree", str(run["degree"]),
        "--mesh", str(run["nx"]), str(run["ny"]),
        "--subdomains", str(run["sx"]), str(run["sy"]),
        "--block-size", str(run["block_size"]),
        "--tdof", str(run["tdof"]),
        "--maxit", str(maxit),
        "--rtol", str(rtol),
        "--output", str(row_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"run_id={run['run_id']} failed\ncmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    row_df = pd.read_csv(row_file)
    if row_df.shape[0] != 1:
        raise RuntimeError(f"expected 1 row in {row_file}, got {row_df.shape[0]}")
    row = row_df.iloc[0].to_dict()
    row["run_id"] = run["run_id"]
    return row


def run_e2e_study(
    exe: Path,
    output_dir: Path,
    degree: int,
    precision: str,
    maxit: int,
    rtol: float,
    max_mesh: int,
    sub_side: int,
) -> pd.DataFrame:
    rows_dir = output_dir / "e2e_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    plan = build_e2e_plan(
        degree=degree, precision=precision, max_mesh=max_mesh,
        sub_side=sub_side, block_size=0, tdof=0,
    )
    print(f"[e2e] Planned runs: {len(plan)}")

    rows, failures = [], []
    t0 = time.time()
    for idx, run in enumerate(plan, start=1):
        row_file = rows_dir / f"run_{run['run_id']:04d}.csv"
        if row_file.exists():
            row_file.unlink()
        print(f"[{idx:02d}/{len(plan):02d}] mesh={run['nx']}x{run['ny']}, sub={run['sx']}x{run['sy']}")
        try:
            rows.append(run_e2e_benchmark(exe=exe, run=run, maxit=maxit, rtol=rtol, row_file=row_file))
        except RuntimeError as err:
            failures.append({"run_id": run["run_id"], "error": str(err)})
            print(f"  -> FAILED: {err}")

    elapsed = time.time() - t0
    print(f"[e2e] {len(rows)} runs in {elapsed / 60:.1f} min; {len(failures)} failures")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "e2e_failures.csv", index=False)
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Kernel plots
# ---------------------------------------------------------------------------


def _add_req_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "requested_block_size" in df.columns:
        df["req_bs"] = df["requested_block_size"].astype(int)
        df["req_td"] = df["requested_tdof"].astype(int)
    else:
        df["req_bs"] = df["kernel_block_size"].astype(int)
        df["req_td"] = df["kernel_tdof"].astype(int)
    return df


SOLVER_STYLE = {
    "WaveHoltz": {"col": "wh_p50_ms", "linestyle": "-", "marker": "o", "color": "tab:blue"},
    "SparseDirect": {"col": "sd_p50_ms", "linestyle": "--", "marker": "s", "color": "tab:orange"},
}

KERNEL_COLORS = {
    (256, 1): "tab:blue",
    (1024, 1): "tab:orange",
    (1024, 4): "tab:green",
}


def plot_kernel_scaling_per_config(df: pd.DataFrame, plot_dir: Path) -> None:
    """Runtime + throughput for each kernel config (WH vs SD per-action)."""
    df = _add_req_cols(df)
    for bs, td in ALL_KERNEL_CONFIGS:
        sub = df[(df["req_bs"] == bs) & (df["req_td"] == td)].sort_values("n_dof")
        if sub.empty:
            continue
        x = sub["n_dof"]

        # Runtime
        fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
        for label, st in SOLVER_STYLE.items():
            ax.plot(x, sub[st["col"]], marker=st["marker"], linestyle=st["linestyle"],
                    linewidth=1.9, markersize=5, color=st["color"], label=label)
        _set_dof_xaxis(ax)
        ax.set_yscale("log", base=10)
        _style_axes(ax)
        ax.set_xlabel("#DOFs")
        ax.set_ylabel("Action time [ms]  (median)")
        ax.set_title(_kernel_title(bs, td))
        ax.legend()
        fig.savefig(plot_dir / f"kernel_runtime_{_kernel_slug(bs, td)}.pdf", bbox_inches="tight")
        plt.close(fig)

        # Throughput
        fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
        for label, st in SOLVER_STYLE.items():
            y = x / (sub[st["col"]] * 1e3)
            ax.plot(x, y, marker=st["marker"], linestyle=st["linestyle"],
                    linewidth=1.9, markersize=5, color=st["color"], label=label)
        _set_dof_xaxis(ax)
        ax.set_ylim(bottom=0)
        _style_axes(ax)
        ax.set_xlabel("#DOFs")
        ax.set_ylabel("Throughput  [MDOFs/s]")
        ax.set_title(_kernel_title(bs, td))
        ax.legend()
        fig.savefig(plot_dir / f"kernel_throughput_{_kernel_slug(bs, td)}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_kernel_selected_configs(df: pd.DataFrame, plot_dir: Path) -> None:
    """Combined runtime and throughput across selected kernel configs."""
    df = _add_req_cols(df)
    sel = df[df[["req_bs", "req_td"]].apply(tuple, axis=1).isin(SELECTED_KERNEL_CONFIGS)].copy()
    if sel.empty:
        return

    for metric, ylabel, yscale, fname_stem in [
        ("runtime", "Action time [ms]  (median)", "log", "kernel_runtime_selected"),
        ("throughput", "Throughput  [MDOFs/s]", "linear", "kernel_throughput_selected"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
        for bs, td in SELECTED_KERNEL_CONFIGS:
            kdata = sel[(sel["req_bs"] == bs) & (sel["req_td"] == td)].sort_values("n_dof")
            if kdata.empty:
                continue
            x = kdata["n_dof"]
            color = KERNEL_COLORS.get((bs, td), "tab:gray")
            klabel = _kernel_title(bs, td)
            for solver, st in SOLVER_STYLE.items():
                y_vals = kdata[st["col"]] if metric == "runtime" else x / (kdata[st["col"]] * 1e3)
                ax.plot(x, y_vals, marker=st["marker"], linestyle=st["linestyle"],
                        linewidth=1.9, markersize=5, color=color, label=f"{solver}, {klabel}")
        _set_dof_xaxis(ax)
        if yscale == "log":
            ax.set_yscale("log", base=10)
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, y1 * 1.4)
        else:
            ax.set_ylim(bottom=0)
            _, y1 = ax.get_ylim()
            ax.set_ylim(0, y1 * 1.2)
        _style_axes(ax)
        ax.set_xlabel("#DOFs")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, ncol=2)
        fig.savefig(plot_dir / f"{fname_stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_kernel_sd_build_stats(df: pd.DataFrame, plot_dir: Path) -> None:
    """SparseDirect factorization time (analysis + factor + total) and factor memory vs. mesh size."""
    df = _add_req_cols(df)
    # Aggregate over all kernel configs (SD build cost doesn't depend on WH kernel config).
    grp = (
        df.groupby("n_dof", as_index=False)
        .agg(
            analysis_s=("sd_analysis_s", "mean"),
            factor_s=("sd_factor_s", "mean"),
            build_s=("sd_build_s", "mean"),
            factor_mib=("sd_factor_mib", "mean"),
        )
        .sort_values("n_dof")
    )
    if grp.empty:
        return
    x = grp["n_dof"]

    # Factorization time breakdown
    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, grp["build_s"] * 1e3, marker="D", linewidth=1.9, markersize=5,
            color="tab:red", label="wall-clock build")
    ax.plot(x, grp["factor_s"] * 1e3, marker="^", linewidth=1.9, markersize=5,
            color="tab:orange", linestyle="--", label="CuDSS factor")
    ax.plot(x, grp["analysis_s"] * 1e3, marker="v", linewidth=1.9, markersize=5,
            color="tab:purple", linestyle=":", label="CuDSS analysis")
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Time [ms]")
    ax.set_title("SparseDirect: factorization cost")
    ax.legend()
    fig.savefig(plot_dir / "kernel_sd_factor_time.pdf", bbox_inches="tight")
    plt.close(fig)

    # Factor memory
    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, grp["factor_mib"], marker="o", linewidth=1.9, markersize=5, color="tab:red")
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Factor memory [MiB]")
    ax.set_title("SparseDirect: LU factor memory")
    fig.savefig(plot_dir / "kernel_sd_factor_memory.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_kernel_sd_action_vs_wh(df: pd.DataFrame, plot_dir: Path) -> None:
    """Ratio sd_p50 / wh_p50 (>1 means WH is faster per action call)."""
    df = _add_req_cols(df)
    for bs, td in SELECTED_KERNEL_CONFIGS:
        sub = df[(df["req_bs"] == bs) & (df["req_td"] == td)].sort_values("n_dof")
        if sub.empty:
            continue
        ratio = sub["sd_p50_ms"] / sub["wh_p50_ms"]
        x = sub["n_dof"]

        fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
        ax.plot(x, ratio, marker="o", linewidth=1.9, markersize=5, color="tab:green")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.2, label="parity")
        _set_dof_xaxis(ax)
        _style_axes(ax)
        ax.set_xlabel("#DOFs")
        ax.set_ylabel("SD action time / WH action time")
        ax.set_title(f"Per-action speedup ratio  {_kernel_title(bs, td)}")
        ax.legend()
        fig.savefig(plot_dir / f"kernel_action_ratio_{_kernel_slug(bs, td)}.pdf", bbox_inches="tight")
        plt.close(fig)


def make_kernel_plots(df: pd.DataFrame, output_dir: Path) -> None:
    configure_plot_style()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_kernel_scaling_per_config(df, plot_dir)
    plot_kernel_selected_configs(df, plot_dir)
    plot_kernel_sd_build_stats(df, plot_dir)
    plot_kernel_sd_action_vs_wh(df, plot_dir)


# ---------------------------------------------------------------------------
# E2E plots
# ---------------------------------------------------------------------------


def plot_e2e_total_time(df: pd.DataFrame, plot_dir: Path) -> None:
    """Total wall-clock time (build + solve) vs. #DOFs for both solvers."""
    df = df.sort_values("n_dof")
    x = df["n_dof"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, df["wh_total_s"], marker="o", linewidth=1.9, markersize=5,
            color="tab:blue", label="WaveHoltz")
    ax.plot(x, df["sd_total_s"], marker="s", linewidth=1.9, markersize=5,
            color="tab:orange", label="SparseDirect")
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Total time [s]  (build + solve)")
    ax.set_title("End-to-end solve time")
    ax.legend()
    fig.savefig(plot_dir / "e2e_total_time.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e2e_build_solve_breakdown(df: pd.DataFrame, plot_dir: Path) -> None:
    """Grouped bar chart: build and solve time for WH vs SD at each mesh size."""
    df = df.sort_values("n_dof").reset_index(drop=True)
    labels = [f"{row.nx}×{row.ny}" for row in df.itertuples()]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7.2, 1.2 * len(labels)), 4.8), layout="constrained")
    ax.bar(x - w / 2, df["wh_build_s"], w / 2, label="WH build", color="tab:blue", alpha=0.55)
    ax.bar(x - w / 2, df["wh_solve_s"], w / 2, bottom=df["wh_build_s"],
           label="WH solve", color="tab:blue", alpha=0.95)
    ax.bar(x + w / 2, df["sd_build_s"], w / 2, label="SD build", color="tab:orange", alpha=0.55)
    ax.bar(x + w / 2, df["sd_solve_s"], w / 2, bottom=df["sd_build_s"],
           label="SD solve", color="tab:orange", alpha=0.95)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Time [s]")
    ax.set_title("Build vs. solve time breakdown")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.savefig(plot_dir / "e2e_build_solve_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e2e_speedup(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    Speedup of SparseDirect over WaveHoltz for total time and solve-only time.
    Ratio > 1 means WaveHoltz is slower (SparseDirect wins).
    """
    df = df.sort_values("n_dof")
    x = df["n_dof"]
    total_speedup = df["wh_total_s"] / df["sd_total_s"]
    solve_speedup = df["wh_solve_s"] / df["sd_solve_s"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, total_speedup, marker="o", linewidth=1.9, markersize=5, color="tab:red",
            label="total (build+solve)")
    ax.plot(x, solve_speedup, marker="s", linewidth=1.9, markersize=5, color="tab:purple",
            linestyle="--", label="solve only")
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=1.2, label="parity")
    _set_dof_xaxis(ax)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("WH time / SD time  (>1 means SD wins)")
    ax.set_title("SparseDirect speedup over WaveHoltz")
    ax.legend()
    fig.savefig(plot_dir / "e2e_speedup.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e2e_iterations(df: pd.DataFrame, plot_dir: Path) -> None:
    """Outer MINRES iteration counts vs. #DOFs for both solvers."""
    df = df.sort_values("n_dof")
    x = df["n_dof"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, df["wh_iterations"], marker="o", linewidth=1.9, markersize=5,
            color="tab:blue", label="WaveHoltz")
    ax.plot(x, df["sd_iterations"], marker="s", linewidth=1.9, markersize=5,
            color="tab:orange", label="SparseDirect")
    _set_dof_xaxis(ax)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Outer MINRES iterations")
    ax.set_title("Iteration count to convergence")
    ax.legend()
    fig.savefig(plot_dir / "e2e_iterations.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e2e_residuals(df: pd.DataFrame, plot_dir: Path) -> None:
    """Final Helmholtz residual |b - Au| / |b| for both solvers."""
    df = df.sort_values("n_dof")
    x = df["n_dof"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, df["wh_residual"], marker="o", linewidth=1.9, markersize=5,
            color="tab:blue", label="WaveHoltz")
    ax.plot(x, df["sd_residual"], marker="s", linewidth=1.9, markersize=5,
            color="tab:orange", label="SparseDirect")
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel(r"$|b - Au| \,/\, |b|$")
    ax.set_title("Final Helmholtz residual")
    ax.legend()
    fig.savefig(plot_dir / "e2e_residuals.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e2e_factor_memory(df: pd.DataFrame, plot_dir: Path) -> None:
    """SparseDirect LU factor memory and factorization time vs. #DOFs."""
    df = df.sort_values("n_dof")
    x = df["n_dof"]

    # Memory
    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, df["sd_factor_mib"], marker="o", linewidth=1.9, markersize=5, color="tab:red")
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Factor memory [MiB]")
    ax.set_title("SparseDirect: LU factor memory (E2E)")
    fig.savefig(plot_dir / "e2e_factor_memory.pdf", bbox_inches="tight")
    plt.close(fig)

    # Factorization timing breakdown
    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, df["sd_build_s"] * 1e3, marker="D", linewidth=1.9, markersize=5,
            color="tab:red", label="wall-clock build")
    ax.plot(x, df["sd_factor_s"] * 1e3, marker="^", linewidth=1.9, markersize=5,
            color="tab:orange", linestyle="--", label="CuDSS factor")
    ax.plot(x, df["sd_analysis_s"] * 1e3, marker="v", linewidth=1.9, markersize=5,
            color="tab:purple", linestyle=":", label="CuDSS analysis")
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Time [ms]")
    ax.set_title("SparseDirect: factorization cost (E2E)")
    ax.legend()
    fig.savefig(plot_dir / "e2e_factor_time.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e2e_sd_action_timing(df: pd.DataFrame, plot_dir: Path) -> None:
    """SparseDirect per-action time (min/avg/max) vs. #DOFs from E2E solve stats."""
    df = df.sort_values("n_dof")
    x = df["n_dof"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.fill_between(x, df["sd_action_min_ms"], df["sd_action_max_ms"],
                    alpha=0.25, color="tab:orange", label="min–max band")
    ax.plot(x, df["sd_action_avg_ms"], marker="o", linewidth=1.9, markersize=5,
            color="tab:orange", label="avg per action")
    ax.plot(x, df["sd_action_min_ms"], marker="v", linewidth=1.0, markersize=4,
            color="tab:orange", linestyle=":", alpha=0.7)
    ax.plot(x, df["sd_action_max_ms"], marker="^", linewidth=1.0, markersize=4,
            color="tab:orange", linestyle=":", alpha=0.7)
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Per-action time [ms]")
    ax.set_title("SparseDirect: action timing within E2E solve")
    ax.legend()
    fig.savefig(plot_dir / "e2e_sd_action_timing.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e2e_cost_per_iter(df: pd.DataFrame, plot_dir: Path) -> None:
    """Effective cost per outer MINRES iteration: solve_time / iterations."""
    df = df.sort_values("n_dof")
    x = df["n_dof"]
    wh_cpi = df["wh_solve_s"] / df["wh_iterations"] * 1e3
    sd_cpi = df["sd_solve_s"] / df["sd_iterations"] * 1e3

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    ax.plot(x, wh_cpi, marker="o", linewidth=1.9, markersize=5, color="tab:blue", label="WaveHoltz")
    ax.plot(x, sd_cpi, marker="s", linewidth=1.9, markersize=5, color="tab:orange", label="SparseDirect")
    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Cost per MINRES iteration [ms]")
    ax.set_title("Effective iteration cost")
    ax.legend()
    fig.savefig(plot_dir / "e2e_cost_per_iter.pdf", bbox_inches="tight")
    plt.close(fig)


def make_e2e_plots(df: pd.DataFrame, output_dir: Path) -> None:
    configure_plot_style()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_e2e_total_time(df, plot_dir)
    plot_e2e_build_solve_breakdown(df, plot_dir)
    plot_e2e_speedup(df, plot_dir)
    plot_e2e_iterations(df, plot_dir)
    plot_e2e_residuals(df, plot_dir)
    plot_e2e_factor_memory(df, plot_dir)
    plot_e2e_sd_action_timing(df, plot_dir)
    plot_e2e_cost_per_iter(df, plot_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    exe = Path(args.exe)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_kernel = args.mode in ("kernel", "all")
    run_e2e = args.mode in ("e2e", "all")

    kernel_csv = output_dir / "compare_sparse2d_kernel.csv"
    e2e_csv = output_dir / "compare_sparse2d_e2e.csv"

    # ── Kernel study ──────────────────────────────────────────────────────────
    if run_kernel:
        if args.skip_runs:
            if not kernel_csv.exists():
                raise FileNotFoundError(f"kernel CSV not found: {kernel_csv}")
            df_kernel = pd.read_csv(kernel_csv)
            print(f"Loaded kernel data from {kernel_csv}  ({len(df_kernel)} rows)")
        else:
            if not exe.exists():
                raise FileNotFoundError(f"executable not found: {exe}")
            df_kernel = run_kernel_study(
                exe=exe,
                output_dir=output_dir,
                degree=args.degree,
                precision=args.precision,
                warmup=args.warmup,
                iterations=args.iterations,
                max_mesh=args.max_kernel_mesh,
            )
            df_kernel.to_csv(kernel_csv, index=False)
            print(f"Saved kernel data to {kernel_csv}")

        make_kernel_plots(df_kernel, output_dir)
        print(f"Kernel plots saved to {output_dir / 'plots'}")

    # ── E2E study ─────────────────────────────────────────────────────────────
    if run_e2e:
        if args.skip_runs:
            if not e2e_csv.exists():
                raise FileNotFoundError(f"E2E CSV not found: {e2e_csv}")
            df_e2e = pd.read_csv(e2e_csv)
            print(f"Loaded E2E data from {e2e_csv}  ({len(df_e2e)} rows)")
        else:
            if not exe.exists():
                raise FileNotFoundError(f"executable not found: {exe}")
            df_e2e = run_e2e_study(
                exe=exe,
                output_dir=output_dir,
                degree=args.degree,
                precision=args.precision,
                maxit=args.maxit,
                rtol=args.rtol,
                max_mesh=args.max_e2e_mesh,
                sub_side=args.e2e_sub,
            )
            df_e2e.to_csv(e2e_csv, index=False)
            print(f"Saved E2E data to {e2e_csv}")

        make_e2e_plots(df_e2e, output_dir)
        print(f"E2E plots saved to {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
