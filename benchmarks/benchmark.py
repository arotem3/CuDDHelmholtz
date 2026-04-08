import argparse
import math
import subprocess
import time
from pathlib import Path

import matplotlib.axes
import matplotlib.lines
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
DEGREES: list[int] = [1, 2, 3, 4, 7]

KERNEL_BLOCK_SIZES: list[int] = [256, 512, 1024]
KERNEL_TDOFS: list[int] = [1, 2, 4]
KERNEL_CONFIGS: list[tuple[int, int]] = [
    (b, k) for b in KERNEL_BLOCK_SIZES for k in KERNEL_TDOFS
]

# Kernel configs to include in the scaling study.
# Automatic kernel selection is intentionally disabled to keep comparisons
# interpretable across the parameter sweep.
SCALING_KERNEL_CONFIGS: list[tuple[int, int]] = KERNEL_CONFIGS


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
    """
    Choose subdomain dimensions from kernel configuration.
    Returns None only when a fixed kernel cannot represent even one element.
    """
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
        description="Run a 2D DDH operator benchmark study and produce plots."
    )
    parser.add_argument(
        "--exe",
        default=str(Path("build") / "benchmarks" / "benchmark2d"),
        help="Path to benchmark2d executable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("benchmarks") / "results" / "ddh2d_study"),
        help="Directory where combined data and figures will be written.",
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Timed iterations per benchmark run."
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations per benchmark run."
    )
    parser.add_argument(
        "--precision",
        choices=["float", "double", "both"],
        default="both",
        help="Precision for baseline studies.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip execution and only regenerate plots from the combined CSV file.",
    )
    return parser.parse_args()


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def nearest_divisor(n: int, target: int) -> int:
    ds = divisors(n)
    return min(ds, key=lambda d: abs(d - target))


def default_subdomains(nx: int, ny: int) -> tuple[int, int]:
    sx = nearest_divisor(nx, max(2, nx // 8))
    sy = nearest_divisor(ny, max(2, ny // 8))
    return sx, sy


def candidate_subdomains(nx: int, ny: int) -> list[tuple[int, int]]:
    targets = [
        (max(2, nx // 4), max(2, ny // 4)),
        (max(2, nx // 8), max(2, ny // 8)),
        (max(2, nx // 16), max(2, ny // 16)),
    ]
    result: list[tuple[int, int]] = []
    for tx, ty in targets:
        sx = nearest_divisor(nx, tx)
        sy = nearest_divisor(ny, ty)
        key = (sx, sy)
        if key not in result:
            result.append(key)
    return result


def scaling_omega(degree: int, nx: int, ny: int) -> float:
    return 0.1 * degree * max(nx, ny)


def build_plan(precision_mode: str) -> list[dict]:
    plan: list[dict] = []
    run_id = 1
    precisions = ["float", "double"] if precision_mode == "both" else [precision_mode]

    # --- Study: degree & mesh scaling under fixed kernel configs ---
    for nx, ny in MESHES:
        for degree in DEGREES:
            for block_size, tdof in SCALING_KERNEL_CONFIGS:
                sub_dims = subdomains_for_kernel(nx, ny, degree, block_size, tdof)
                if sub_dims is None:
                    continue
                sx, sy = sub_dims
                for precision in precisions:
                    plan.append(
                        {
                            "run_id": run_id,
                            "study": "degree_mesh_scaling",
                            "precision": precision,
                            "degree": degree,
                            "nx": nx,
                            "ny": ny,
                            "sx": sx,
                            "sy": sy,
                            "block_size": block_size,
                            "tdof": tdof,
                            "omega": scaling_omega(degree, nx, ny),
                        }
                    )
                    run_id += 1

    return plan


def run_benchmark(
    exe: Path, run: dict, warmup: int, iterations: int, row_file: Path
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
        "--omega",
        str(run["omega"]),
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
            f"benchmark2d failed for run_id={run['run_id']}\n"
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
            "study": run["study"],
            "requested_block_size": run["block_size"],
            "requested_tdof": run["tdof"],
        }
    )
    return row


def run_study(
    exe: Path,
    output_dir: Path,
    warmup: int,
    iterations: int,
    precision_mode: str,
) -> pd.DataFrame:
    rows_dir = output_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(precision_mode)
    print(f"Planned runs: {len(plan)}")

    rows = []
    failures = []
    t0 = time.time()
    for idx, run in enumerate(plan, start=1):
        row_file = rows_dir / f"run_{run['run_id']:04d}.csv"
        if row_file.exists():
            row_file.unlink()

        print(
            f"[{idx:03d}/{len(plan):03d}] study={run['study']}, precision={run['precision']}, "
            f"degree={run['degree']}, mesh={run['nx']}x{run['ny']}, sub={run['sx']}x{run['sy']}, "
            f"block={run['block_size']}, tdof={run['tdof']}, omega={run['omega']}"
        )
        try:
            row = run_benchmark(
                exe, run, warmup=warmup, iterations=iterations, row_file=row_file
            )
            rows.append(row)
        except RuntimeError as err:
            failures.append(
                {"run_id": run["run_id"], "study": run["study"], "error": str(err)}
            )
            print(f"  -> FAILED run_id={run['run_id']}: {err}")

    elapsed = time.time() - t0
    print(f"Completed {len(rows)} successful runs in {elapsed / 60.0:.1f} minutes")
    if failures:
        print(f"Failed runs: {len(failures)}")
        pd.DataFrame(failures).to_csv(
            output_dir / "benchmark2d_failures.csv", index=False
        )

    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    return df


def save_combined_data(df: pd.DataFrame, output_dir: Path) -> Path:
    combined_csv = output_dir / "benchmark2d_combined.csv"
    df.to_csv(combined_csv, index=False)
    return combined_csv


def ensure_plot_dir(output_dir: Path) -> Path:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _kernel_title(block_size: int, tdof: int) -> str:
    """Human-readable fixed kernel configuration label."""
    return f"$B = {block_size}, T = {tdof}$"


def _kernel_slug(block_size: int, tdof: int) -> str:
    return f"B{block_size}_T{tdof}"


def _kernel_columns(sub: pd.DataFrame) -> pd.DataFrame:
    if "requested_block_size" in sub.columns:
        sub = sub.copy()
        sub["req_bs"] = sub["requested_block_size"].astype(int)
        sub["req_td"] = sub["requested_tdof"].astype(int)
        return sub

    sub = sub.copy()
    sub["req_bs"] = sub["kernel_block_size"].astype(int)
    sub["req_td"] = sub["kernel_tdof"].astype(int)
    return sub


def _single_precision(sub: pd.DataFrame) -> pd.DataFrame:
    return sub[sub["precision"] == "float"].copy()


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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scaling_per_kernel(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    Produce per-kernel scaling figures (single precision):
      1) absolute runtime [ms]
      2) runtime / helmholtz runtime (log y-axis)
            3) throughput [MDOF/s]

        x-axis: number of degrees of freedom (log-10 scale).
    lines: polynomial degree.
    """
    sub = df[df["study"] == "degree_mesh_scaling"].copy()
    if sub.empty:
        return

    sub = _kernel_columns(sub)
    sub = _single_precision(sub)
    sub["throughput"] = sub["n_dof"] / sub["avg_ms"] * 1e-3  # MDOF/s

    degrees = sorted(sub["degree"].unique())
    palette = [plt.get_cmap("tab10")(i) for i in range(len(degrees))]
    degree_colors = dict(zip(degrees, palette))
    for bs, td in KERNEL_CONFIGS:
        kdata = sub[(sub["req_bs"] == bs) & (sub["req_td"] == td)]
        if kdata.empty:
            continue

        for metric, ylabel, suffix, log_y in [
            ("avg_ms", "Runtime [ms]", "runtime", False),
            ("avg_rel_to_helmholtz", "DD Time / Operator Time", "relative", False),
            ("throughput", "Throughput [MDOF/s]", "throughput", False),
        ]:
            fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
            for deg in degrees:
                ddata = kdata[kdata["degree"] == deg].sort_values("n_dof")
                if ddata.empty:
                    continue
                ax.plot(
                    ddata["n_dof"],
                    ddata[metric],
                    marker="o",
                    linewidth=1.9,
                    markersize=5,
                    color=degree_colors[deg],
                    label=f"$p={deg}$",
                )

            _set_dof_xaxis(ax)
            if log_y:
                ax.set_yscale("log", base=10)
                ax.set_ylim(bottom=1)
            else:
                ax.set_ylim(bottom=0)

            if suffix == "relative":
                ax.set_ylim(top=50)
                ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
                ax.text(0.95, 1.05, r"$1\times$", transform=ax.get_yaxis_transform(), ha="right", va="bottom")

            _style_axes(ax)
            ax.set_xlabel("#DOFs")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{_kernel_title(bs, td)}")
            ax.legend(title="Degree")
            fig.savefig(
                plot_dir / f"scaling_{suffix}_{_kernel_slug(bs, td)}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_fixed_degree_kernel_comparison(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    For fixed degree (1 and 3), compare all kernels together as mesh grows.
    Produce both runtime and throughput figures.
    """
    sub = df[df["study"] == "degree_mesh_scaling"].copy()
    if sub.empty:
        return

    sub = _kernel_columns(sub)
    sub = _single_precision(sub)
    sub["throughput"] = sub["n_dof"] / sub["avg_ms"] * 1e-3  # MDOF/s

    degree_targets = [1, 3, 7]
    palette = [plt.get_cmap("tab10")(i) for i in range(len(KERNEL_CONFIGS))]
    kernel_colors = dict(zip(KERNEL_CONFIGS, palette))

    for deg in degree_targets:
        dsub = sub[sub["degree"] == deg]
        if dsub.empty:
            continue

        for metric, ylabel, slug in [
            ("avg_ms", "Runtime [ms]", "runtime"),
            ("throughput", "Throughput [MDOF/s]", "throughput"),
        ]:
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            for bs, td in KERNEL_CONFIGS:
                kdata = dsub[
                    (dsub["req_bs"] == bs) & (dsub["req_td"] == td)
                ].sort_values("n_dof")
                if kdata.empty:
                    continue
                ax.plot(
                    kdata["n_dof"],
                    kdata[metric],
                    marker="o",
                    linewidth=1.9,
                    markersize=5,
                    color=kernel_colors[(bs, td)],
                    label=_kernel_title(bs, td),
                )

            _set_dof_xaxis(ax)
            ax.set_ylim(bottom=0)
            if metric == 'throughput':
                ax.set_ylim(top=550)

            _style_axes(ax)
            ax.set_xlabel("#DOFs")
            ax.set_ylabel(ylabel)
            ax.legend(title="Kernel", fontsize=9)
            fig.tight_layout()
            fig.savefig(
                plot_dir / f"kernel_comparison_degree_{deg}_{slug}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_precision_ratio(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    Double / single runtime ratio for a fixed kernel B=1024, K=1.
    x-axis: number of degrees of freedom (log-10 scale).
    Lines:  polynomial degree.
    """
    sub = df[df["study"] == "degree_mesh_scaling"].copy()
    if sub.empty:
        return

    sub = _kernel_columns(sub)
    sub = sub[(sub["req_bs"] == 1024) & (sub["req_td"] == 1)]

    pivot = (
        sub.groupby(["nx", "ny", "degree", "n_dof", "precision"], as_index=False)[
            "avg_ms"
        ]
        .mean()
        .pivot(
            index=["nx", "ny", "degree", "n_dof"], columns="precision", values="avg_ms"
        )
        .dropna()
        .reset_index()
    )
    if "float" not in pivot.columns or "double" not in pivot.columns:
        return

    pivot["ratio"] = pivot["double"] / pivot["float"]

    degrees = sorted(pivot["degree"].unique())
    palette = [plt.get_cmap("tab10")(i) for i in range(len(degrees))]
    degree_colors = dict(zip(degrees, palette))

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for deg in degrees:
        g = pivot[pivot["degree"] == deg].sort_values("n_dof")
        ax.plot(
            g["n_dof"],
            g["ratio"],
            marker="^",
            linewidth=1.8,
            markersize=5,
            color=degree_colors[deg],
            label=f"$p={deg}$",
        )

    _set_dof_xaxis(ax)
    _style_axes(ax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Runtime ratio (double / single)")
    ax.set_title(r"Precision overhead ($B=1024, K=1$)")
    ax.legend(title="Degree")
    fig.tight_layout()
    fig.savefig(plot_dir / "precision_ratio.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    configure_plot_style()
    plot_dir = ensure_plot_dir(output_dir)
    plot_scaling_per_kernel(df, plot_dir)
    plot_fixed_degree_kernel_comparison(df, plot_dir)
    plot_precision_ratio(df, plot_dir)
    plot_helmholtz_throughput(df, plot_dir)


def plot_helmholtz_throughput(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    Throughput of the standalone Helmholtz operator (single precision).
    The Helmholtz runtime is independent of the DDH kernel config, so results
    are averaged across all kernel configs before plotting.

    x-axis: number of degrees of freedom (log-10 scale).
    Lines:  polynomial degree.
    """
    sub = df[df["study"] == "degree_mesh_scaling"].copy()
    if sub.empty or "helmholtz_avg_ms" not in sub.columns:
        return

    sub = _single_precision(sub)

    # Average over kernel configs (Helmholtz is kernel-independent)
    agg = sub.groupby(["nx", "ny", "degree", "n_dof"], as_index=False)[
        "helmholtz_avg_ms"
    ].mean()
    agg["throughput"] = agg["n_dof"] / agg["helmholtz_avg_ms"] * 1e-3  # MDOF/s

    degrees = sorted(agg["degree"].unique())
    palette = [plt.get_cmap("tab10")(i) for i in range(len(degrees))]
    degree_colors = dict(zip(degrees, palette))

    fig, ax = plt.subplots(figsize=(6.6, 4.8), layout="constrained")
    for deg in degrees:
        g = agg[agg["degree"] == deg].sort_values("n_dof")
        ax.plot(
            g["n_dof"],
            g["throughput"],
            marker="s",
            linewidth=1.9,
            markersize=5,
            color=degree_colors[deg],
            label=f"$p={deg}$",
        )

    _set_dof_xaxis(ax)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    ax.set_xlabel("#DOFs")
    ax.set_ylabel("Throughput [MDOF/s]")
    ax.set_title("Helmholtz operator throughput (single precision)")
    ax.legend(title="Degree")
    fig.savefig(plot_dir / "helmholtz_throughput.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exe = Path(args.exe)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = output_dir / "benchmark2d_combined.csv"
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
            warmup=args.warmup,
            iterations=args.iterations,
            precision_mode=args.precision,
        )
        saved_csv = save_combined_data(df, output_dir)
        print(f"Saved combined benchmark data to: {saved_csv}")

    make_plots(df, output_dir)
    print(f"Saved plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
