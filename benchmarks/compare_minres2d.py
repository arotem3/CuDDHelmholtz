import argparse
import math
import subprocess
import time
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


MESHES: list[tuple[int, int]] = [
    (32, 32),
    (64, 64),
    (96, 96),
    (128, 128),
    (192, 192),
    (256, 256),
    (384, 384),
    (512, 512),
    (768, 768),
    (1024, 1024),
]

KERNEL_BLOCK_SIZES: list[int] = [256, 512, 1024]
KERNEL_TDOFS: list[int] = [1, 2, 4]
KERNEL_CONFIGS: list[tuple[int, int]] = [
    (b, k) for b in KERNEL_BLOCK_SIZES for k in KERNEL_TDOFS
]
SELECTED_KERNEL_CONFIGS: list[tuple[int, int]] = [(256, 1), (1024, 1), (1024, 4)]


# ---------------------------------------------------------------------------
# Kernel-driven subdomain sizing
# ---------------------------------------------------------------------------


def kernel_capacity_elements(degree: int, block_size: int, tdof: int) -> int:
    """
    Return target per-subdomain element capacity E = sx*sy implied by kernel:
    E = floor(B / P^2) * K.
    """
    p2 = (degree + 1) * (degree + 1)
    return (block_size // p2) * tdof


def choose_subdomain_dims(nx: int, ny: int, target_elems: int) -> tuple[int, int]:
    """
    Choose sx,sy (subdomain element dimensions) so that sx*sy <= target_elems,
    maximizing fill while preserving the mesh aspect ratio as much as possible.
    """
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

        aspect = sx / sy
        aspect_err = abs(math.log(aspect / target_aspect))

        if fill > best_fill or aspect_err < best_aspect_err:
            best_sx, best_sy = sx, sy
            best_fill = fill
            best_aspect_err = aspect_err

    return best_sx, best_sy


def subdomains_for_kernel(
    nx: int, ny: int, degree: int, block_size: int, tdof: int
) -> tuple[int, int] | None:
    target_elems = kernel_capacity_elements(degree, block_size, tdof)
    if target_elems <= 0:
        return None
    return choose_subdomain_dims(nx, ny, target_elems)


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
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep compare_minres2d and plot WaveHoltz vs MINRES runtime scaling."
    )
    parser.add_argument(
        "--exe",
        default=str(Path("build") / "benchmarks" / "compare_minres2d"),
        help="Path to compare_minres2d executable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("benchmarks") / "results" / "compare_minres2d"),
        help="Directory where combined data and figures will be written.",
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="Fixed polynomial degree."
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Timed iterations per benchmark run."
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations per benchmark run."
    )
    parser.add_argument(
        "--precision",
        choices=["float", "double"],
        default="float",
        help="Scalar precision for the DD operator benchmark.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip execution and only regenerate plots from the combined CSV file.",
    )
    return parser.parse_args()


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
    return f"$B = {block_size}, T = {tdof}$"


def _kernel_slug(block_size: int, tdof: int) -> str:
    return f"B{block_size}_T{tdof}"


def build_plan(degree: int, precision: str) -> list[dict]:
    plan: list[dict] = []
    run_id = 1

    for nx, ny in MESHES:
        for block_size, tdof in KERNEL_CONFIGS:
            sub_dims = subdomains_for_kernel(nx, ny, degree, block_size, tdof)
            if sub_dims is None:
                continue
            sx, sy = sub_dims
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


def run_benchmark(
    exe: Path,
    run: dict,
    warmup: int,
    iterations: int,
    row_file: Path,
) -> dict:
    cmd = [
        str(exe),
        "--precision",
        str(run["precision"]),
        "--degree",
        str(run["degree"]),
        "--mesh",
        str(run["nx"]),
        str(run["ny"]),
        "--subdomains",
        str(run["sx"]),
        str(run["sy"]),
        "--block-size",
        str(run["block_size"]),
        "--tdof",
        str(run["tdof"]),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--output",
        str(row_file),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"compare_minres2d failed for run_id={run['run_id']}\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    row_df = pd.read_csv(row_file)
    if row_df.shape[0] != 1:
        raise RuntimeError(f"expected one row in {row_file}, got {row_df.shape[0]}")

    row = row_df.iloc[0].to_dict()
    row.update(
        {
            "run_id": run["run_id"],
            "requested_block_size": run["block_size"],
            "requested_tdof": run["tdof"],
        }
    )
    return row


def run_study(
    exe: Path,
    output_dir: Path,
    degree: int,
    precision: str,
    warmup: int,
    iterations: int,
) -> pd.DataFrame:
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(degree=degree, precision=precision)
    print(f"Planned runs: {len(plan)}")

    rows = []
    failures = []
    t0 = time.time()
    for idx, run in enumerate(plan, start=1):
        row_file = rows_dir / f"run_{run['run_id']:04d}.csv"
        if row_file.exists():
            row_file.unlink()

        print(
            f"[{idx:03d}/{len(plan):03d}] precision={run['precision']}, degree={run['degree']}, "
            f"mesh={run['nx']}x{run['ny']}, sub={run['sx']}x{run['sy']}, "
            f"block={run['block_size']}, tdof={run['tdof']}"
        )

        try:
            row = run_benchmark(
                exe=exe,
                run=run,
                warmup=warmup,
                iterations=iterations,
                row_file=row_file,
            )
            rows.append(row)
        except RuntimeError as err:
            failures.append({"run_id": run["run_id"], "error": str(err)})
            print(f"  -> FAILED run_id={run['run_id']}: {err}")

    elapsed = time.time() - t0
    print(f"Completed {len(rows)} successful runs in {elapsed / 60.0:.1f} minutes")
    if failures:
        print(f"Failed runs: {len(failures)}")
        pd.DataFrame(failures).to_csv(
            output_dir / "compare_minres2d_failures.csv", index=False
        )

    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def ensure_plot_dir(output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def plot_scaling_per_kernel(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    For each fixed kernel config, plot runtime and throughput vs DOFs for both
    subdomain solvers.
    """
    if df.empty:
        return

    required = {
        "n_dof",
        "waveholtz_p10_ms",
        "waveholtz_p50_ms",
        "waveholtz_p90_ms",
        "minres_p10_ms",
        "minres_p50_ms",
        "minres_p90_ms",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Missing quantile columns in combined CSV: " + ", ".join(missing)
        )

    sub = df.copy()
    if "requested_block_size" in sub.columns:
        sub["req_bs"] = sub["requested_block_size"].astype(int)
        sub["req_td"] = sub["requested_tdof"].astype(int)
    else:
        sub["req_bs"] = sub["kernel_block_size"].astype(int)
        sub["req_td"] = sub["kernel_tdof"].astype(int)

    for bs, td in KERNEL_CONFIGS:
        kdata = sub[(sub["req_bs"] == bs) & (sub["req_td"] == td)].sort_values("n_dof")
        if kdata.empty:
            continue

        x = kdata["n_dof"]

        wh_y = kdata["waveholtz_p50_ms"]
        mr_y = kdata["minres_p50_ms"]

        # Runtime plot (log y-scale)
        fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
        ax.plot(
            x,
            wh_y,
            marker="o",
            linewidth=1.9,
            markersize=5,
            color="tab:blue",
            label="WaveHoltz",
        )
        ax.plot(
            x,
            mr_y,
            marker="s",
            linewidth=1.9,
            markersize=5,
            color="tab:orange",
            label="MINRES",
        )

        _set_dof_xaxis(ax)
        ax.set_yscale("log", base=10)
        _style_axes(ax)
        ax.set_xlabel("#DOFs")
        ax.set_ylabel("Runtime [ms]")
        ax.set_title(_kernel_title(bs, td))
        ax.legend()
        fig.savefig(
            plot_dir / f"compare_minres2d_runtime_{_kernel_slug(bs, td)}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        # Throughput plot (MDOFs/s)
        wh_thr_p50 = x / (kdata["waveholtz_p50_ms"] * 1e3)
        mr_thr_p50 = x / (kdata["minres_p50_ms"] * 1e3)

        fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
        ax.plot(
            x,
            wh_thr_p50,
            marker="o",
            linewidth=1.9,
            markersize=5,
            color="tab:blue",
            label="WaveHoltz",
        )
        ax.plot(
            x,
            mr_thr_p50,
            marker="s",
            linewidth=1.9,
            markersize=5,
            color="tab:orange",
            label="MINRES",
        )

        _set_dof_xaxis(ax)
        ax.set_ylim(bottom=0)
        _style_axes(ax)
        ax.set_xlabel("#DOFs")
        ax.set_ylabel("Throughput [MDOFs/s]")
        ax.set_title(_kernel_title(bs, td))
        ax.legend()
        fig.savefig(
            plot_dir / f"compare_minres2d_throughput_{_kernel_slug(bs, td)}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_selected_kernel_comparison(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    Plot selected kernel configurations in a single figure for runtime and
    throughput. Kernel config is encoded by color, solver by linestyle/marker.
    """
    if df.empty:
        return

    required = {
        "n_dof",
        "waveholtz_p50_ms",
        "minres_p50_ms",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Missing required columns in combined CSV: " + ", ".join(missing)
        )

    sub = df.copy()
    if "requested_block_size" in sub.columns:
        sub["req_bs"] = sub["requested_block_size"].astype(int)
        sub["req_td"] = sub["requested_tdof"].astype(int)
    else:
        sub["req_bs"] = sub["kernel_block_size"].astype(int)
        sub["req_td"] = sub["kernel_tdof"].astype(int)

    selected = sub[
        sub[["req_bs", "req_td"]].apply(tuple, axis=1).isin(SELECTED_KERNEL_CONFIGS)
    ].copy()
    if selected.empty:
        return

    kernel_colors = {
        (256, 1): "tab:blue",
        (1024, 1): "tab:orange",
        (1024, 4): "tab:green",
    }
    solver_style = {
        "WaveHoltz": {"linestyle": "-", "marker": "o", "ycol": "waveholtz_p50_ms"},
        "MINRES": {"linestyle": "--", "marker": "s", "ycol": "minres_p50_ms"},
    }

    # Combined runtime plot
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    for bs, td in SELECTED_KERNEL_CONFIGS:
        kdata = selected[
            (selected["req_bs"] == bs) & (selected["req_td"] == td)
        ].sort_values("n_dof")
        if kdata.empty:
            continue
        x = kdata["n_dof"]
        color = kernel_colors.get((bs, td), "tab:gray")
        kernel_lbl = _kernel_title(bs, td)

        for solver, style in solver_style.items():
            y = kdata[style["ycol"]]
            ax.plot(
                x,
                y,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                markersize=5,
                color=color,
                label=f"{solver}, {kernel_lbl}",
            )

    _set_dof_xaxis(ax)
    ax.set_yscale("log", base=10)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.35)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Runtime [ms]")
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(
        plot_dir / "compare_minres2d_runtime_selected_kernels.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Combined throughput plot
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    for bs, td in SELECTED_KERNEL_CONFIGS:
        kdata = selected[
            (selected["req_bs"] == bs) & (selected["req_td"] == td)
        ].sort_values("n_dof")
        if kdata.empty:
            continue
        x = kdata["n_dof"]
        color = kernel_colors.get((bs, td), "tab:gray")
        kernel_lbl = _kernel_title(bs, td)

        for solver, style in solver_style.items():
            y = x / (kdata[style["ycol"]] * 1e3)
            ax.plot(
                x,
                y,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.9,
                markersize=5,
                color=color,
                label=f"{solver}, {kernel_lbl}",
            )

    _set_dof_xaxis(ax)
    ax.set_ylim(bottom=0)
    _, y1 = ax.get_ylim()
    ax.set_ylim(0, y1 * 1.20)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Throughput [MDOFs/s]")
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(
        plot_dir / "compare_minres2d_throughput_selected_kernels.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    configure_plot_style()
    plot_dir = ensure_plot_dir(output_dir)
    plot_scaling_per_kernel(df, plot_dir)
    plot_selected_kernel_comparison(df, plot_dir)


def main() -> None:
    args = parse_args()
    exe = Path(args.exe)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = output_dir / "compare_minres2d_combined.csv"
    if args.skip_runs:
        if not combined_csv.exists():
            raise FileNotFoundError(f"combined CSV not found: {combined_csv}")
        df = pd.read_csv(combined_csv)
        print(f"Loaded existing data: {combined_csv}")
    else:
        if not exe.exists():
            raise FileNotFoundError(f"benchmark executable not found: {exe}")
        df = run_study(
            exe=exe,
            output_dir=output_dir,
            degree=args.degree,
            precision=args.precision,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        df.to_csv(combined_csv, index=False)
        print(f"Saved combined benchmark data to: {combined_csv}")

    make_plots(df, output_dir)
    print(f"Saved plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
