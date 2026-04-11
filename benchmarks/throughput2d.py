"""
throughput2d.py – ncu speed-of-light profiling of ddh_action_kernel (2-D).

Runs the benchmark2d executable under ncu for nx=ny=512, p=3, single precision
across all combinations of block_size in {256,512,1024} and tdof in {1,2,4}.
Collects:
  - Compute (SM) throughput          [% of peak]
  - Memory throughput                [% of peak]
  - DRAM throughput                  [% of peak]
  - L1 throughput                    [% of peak]
  - L2 throughput                    [% of peak]
  - Achieved occupancy               [% of peak]
  - Theoretical occupancy            [% of peak]
  - FP32 instruction counts          [fadd, fmul, ffma]
  - DRAM bytes transferred           [bytes]
  - Kernel duration                  [ns]
Then plots a grouped bar chart, a roofline model, and saves a summary CSV.
"""

import argparse
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NX = NY = 1024
DEGREE = 3
OMEGA = 0.1 * DEGREE * max(NX, NY)
PRECISION = "float"

BLOCK_SIZES = [256, 512, 1024]
TDOFS = [1, 2, 4]

WARMUP = 3  # kernel launches to skip (--launch-skip)
ITERATIONS = 5  # kernel launches to profile (--launch-count)

# ncu metric IDs for the "speed of light" stats we care about
NCU_METRICS = {
    "compute_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "achieved_occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occ": "sm__maximum_warps_per_active_cycle_pct",
    # Roofline: FP32 instruction counts, DRAM bytes, kernel duration
    "flops_fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "flops_fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "flops_ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "dram_bytes": "dram__bytes.sum",
    "duration_ns": "gpu__time_duration.sum",
}

METRIC_LABELS = {
    "compute_pct": "Compute\n(SM)",
    "memory_pct": "Memory\n(Overall)",
    "dram_pct": "DRAM",
    "l1_pct": "L1",
    "l2_pct": "L2",
    "achieved_occ": "Achieved\nOccupancy",
    "theoretical_occ": "Theoretical\nOccupancy",
}


# ---------------------------------------------------------------------------
# Subdomain sizing  (mirrors logic in benchmark.py)
# ---------------------------------------------------------------------------


def kernel_capacity_elements(degree: int, block_size: int, tdof: int) -> int:
    p2 = (degree + 1) * (degree + 1)
    return (block_size // p2) * tdof


def choose_subdomain_dims(nx: int, ny: int, target_elems: int) -> tuple[int, int]:
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


def subdomain_dims(block_size: int, tdof: int) -> tuple[int, int]:
    cap = kernel_capacity_elements(DEGREE, block_size, tdof)
    return choose_subdomain_dims(NX, NY, cap)


# ---------------------------------------------------------------------------
# ncu invocation and parsing
# ---------------------------------------------------------------------------


def build_ncu_cmd(
    ncu: str,
    exe: Path,
    block_size: int,
    tdof: int,
    sx: int,
    sy: int,
    ncu_out: Path,
) -> list[str]:
    metrics_str = ",".join(NCU_METRICS.values())
    return [
        ncu,
        # target the ddh_action_kernel template instantiations
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        "regex:ddh_action_kernel",
        # skip warmup launches, profile timed launches
        "--launch-skip",
        str(WARMUP),
        "--launch-count",
        str(ITERATIONS),
        # output
        "--metrics",
        metrics_str,
        "--csv",
        "--log-file",
        str(ncu_out),
        "--",
        str(exe),
        "--precision",
        PRECISION,
        "--degree",
        str(DEGREE),
        "--mesh",
        str(NX),
        str(NY),
        "--subdomains",
        str(sx),
        str(sy),
        "--omega",
        str(OMEGA),
        "--block-size",
        str(block_size),
        "--tdof",
        str(tdof),
        "--warmup",
        str(WARMUP),
        "--iterations",
        str(ITERATIONS),
        "--waveholtz-iterations",
        "5",
    ]


def parse_ncu_csv(csv_text: str) -> dict[str, float]:
    """
    Parse ncu --csv output (which goes to the log file).
    Returns a dict mapping our short metric key -> mean value across profiled launches.
    """
    # Build reverse map: ncu_metric_id -> short key
    reverse = {v: k for k, v in NCU_METRICS.items()}

    # ncu prepends some comment lines starting with "==" — strip them
    clean_lines = [l for l in csv_text.splitlines() if not l.startswith("==")]
    clean_text = "\n".join(clean_lines)

    reader = csv.DictReader(io.StringIO(clean_text))

    # Accumulate per-metric values across kernel invocations
    values: dict[str, list[float]] = {k: [] for k in NCU_METRICS}

    for row in reader:
        metric_name = row.get("Metric Name", "").strip()
        if metric_name not in reverse:
            continue
        key = reverse[metric_name]
        raw = row.get("Metric Value", "").strip().replace(",", "")
        try:
            values[key].append(float(raw))
        except ValueError:
            pass

    result: dict[str, float] = {}
    for key, vals in values.items():
        if vals:
            result[key] = float(np.mean(vals))
        else:
            result[key] = float("nan")
    return result


def run_ncu(
    ncu: str,
    exe: Path,
    block_size: int,
    tdof: int,
    output_dir: Path,
) -> dict[str, float]:
    sx, sy = subdomain_dims(block_size, tdof)
    slug = f"B{block_size}_T{tdof}"
    ncu_out = output_dir / f"ncu_{slug}.log"

    cmd = build_ncu_cmd(ncu, exe, block_size, tdof, sx, sy, ncu_out)

    print(f"  Running ncu for {slug}  (subdomain {sx}x{sy}) ...", flush=True)
    print(f"    cmd: {' '.join(cmd)}", flush=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"LC_NUMERIC": "C", **__import__("os").environ},
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ncu failed for {slug}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not ncu_out.exists():
        raise FileNotFoundError(f"ncu log file not created: {ncu_out}")

    csv_text = ncu_out.read_text()
    metrics = parse_ncu_csv(csv_text)
    metrics["block_size"] = float(block_size)
    metrics["tdof"] = float(tdof)
    metrics["sx"] = float(sx)
    metrics["sy"] = float(sy)
    return metrics


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def configure_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "legend.framealpha": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def config_label(block_size: int, tdof: int) -> str:
    return f"$B={block_size}$\n$T={tdof}$"


def _get_vals(df: pd.DataFrame, configs: list, key: str) -> list[float]:
    result = []
    for bs, td in configs:
        sub = df[(df["block_size"] == bs) & (df["tdof"] == td)]
        result.append(float(sub[key].iloc[0]) if len(sub) > 0 else float("nan"))
    return result


def _pynvml_query() -> dict:
    """
    Query the first GPU via pynvml.  Returns a dict with:
      name              str   GPU product name
      peak_dram_gbs     float peak DRAM bandwidth (GB/s)
      peak_fp32_gflops  float peak FP32 throughput (GFLOP/s)
    Returns an empty dict on any failure.
    """
    try:
        import pynvml  # type: ignore[import]  # provided by nvidia-ml-py

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        sm_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        cuda_cores = pynvml.nvmlDeviceGetNumGpuCores(handle)
        pynvml.nvmlShutdown()
        # DDR: factor of 2 for double data rate
        peak_bw = 2.0 * mem_clock_mhz * 1e6 * (bus_width_bits / 8) / 1e9
        # Each CUDA core does 2 FLOPs per cycle (FMA = 1 multiply + 1 add)
        peak_fp32 = cuda_cores * 2 * sm_clock_mhz * 1e6 / 1e9
        return {"name": name, "peak_dram_gbs": peak_bw, "peak_fp32_gflops": peak_fp32}
    except Exception:
        return {}


def query_peak_dram_bw_gbs() -> float | None:
    """
    Return peak theoretical DRAM bandwidth in GB/s via pynvml.
    Returns None if pynvml is unavailable.
    """
    info = _pynvml_query()
    return info.get("peak_dram_gbs")


def compute_roofline_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment df with per-config roofline quantities derived from ncu metrics:
      - flops       : total FP32 operations per kernel launch
      - gflops_s    : achieved FP32 throughput (GFLOP/s)
      - dram_gbs    : achieved DRAM bandwidth (GB/s)
      - arith_int   : arithmetic intensity (FLOP/Byte)
      - peak_dram_gbs_ncu   : peak DRAM BW inferred from dram_pct + achieved BW
      - peak_fp32_gflops_ncu: peak FP32 inferred from compute_pct + achieved GFLOP/s
    """
    df = df.copy()
    flops = df["flops_fadd"] + df["flops_fmul"] + 2.0 * df["flops_ffma"]
    duration_s = df["duration_ns"] * 1e-9

    df["flops"] = flops
    df["gflops_s"] = flops / duration_s / 1e9
    df["dram_gbs"] = df["dram_bytes"] / duration_s / 1e9
    df["arith_int"] = flops / df["dram_bytes"]

    # Infer hardware ceilings: achieved / (pct / 100)
    df["peak_dram_gbs_ncu"] = df["dram_gbs"] / (df["dram_pct"] / 100.0)
    df["peak_fp32_gflops_ncu"] = df["gflops_s"] / (df["compute_pct"] / 100.0)

    return df


def plot_roofline(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    Roofline model: arithmetic intensity (FLOP/Byte) vs performance (GFLOP/s).

    Hardware ceilings:
      - Peak DRAM BW  from pynvml (memory clock × bus width); falls back to
        median of ncu-derived per-config estimates.
      - Peak FP32     median of ncu-derived estimates (achieved / compute_pct%).

    Both ceiling values are printed on the figure as labels.
    """
    df = compute_roofline_data(df)

    # ------------------------------------------------------------------
    # Hardware limits and GPU name
    # ------------------------------------------------------------------
    gpu_info = _pynvml_query()
    peak_dram_gbs = gpu_info.get("peak_dram_gbs") or float(
        df["peak_dram_gbs_ncu"].median()
    )
    # Prefer hardware-derived peak FP32 (cuda_cores × 2 × SM_clock) over the
    # ncu-inferred estimate, which uses SM throughput % and underestimates
    # when the kernel issues many non-FP32 instructions.
    peak_fp32_gflops = gpu_info.get("peak_fp32_gflops") or float(
        df["peak_fp32_gflops_ncu"].median()
    )

    gpu_name = gpu_info.get("name")
    title = f"Roofline Model — {gpu_name}" if gpu_name else "Roofline Model"

    # ------------------------------------------------------------------
    # Axis limits
    # x: from just below min AI to a bit past the ridge
    # ------------------------------------------------------------------
    all_ai = df["arith_int"].dropna()
    ridge_ai = peak_fp32_gflops / peak_dram_gbs  # FLOP/Byte

    log_ai_lo = np.floor(np.log10(all_ai.min())) - 1
    log_ridge = np.log10(ridge_ai)

    log_ai_hi = log_ridge + (log_ridge - log_ai_lo) * 2
    ai_lo = 10**log_ai_lo
    ai_hi = 10**log_ai_hi

    # y lower bound: base on actual data so points below the theoretical memory
    # line (from incomplete DRAM utilization) are never clipped.
    all_gflops = df["gflops_s"].dropna()
    perf_lo = 10 ** (np.floor(np.log10(all_gflops.min())) - 0.3)
    perf_hi = 10 ** (np.ceil(np.log10(peak_fp32_gflops)) + 0.3)

    # ------------------------------------------------------------------
    # Roof line segments
    # ------------------------------------------------------------------
    ai_mem = np.array([ai_lo, ridge_ai])
    perf_mem = peak_dram_gbs * ai_mem

    ai_cmp = np.array([ridge_ai, ai_hi])
    perf_cmp = np.full_like(ai_cmp, peak_fp32_gflops)

    # ------------------------------------------------------------------
    # Plot — draw all data artists first, then force layout, then add labels
    # so that (a) limits are not re-opened by new artists after canvas.draw()
    # and (b) transData reflects the finalised axes geometry when we compute
    # the slope angle.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 5.5), layout="constrained")
    palette = plt.get_cmap("tab10")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_lo, ai_hi)
    ax.set_ylim(perf_lo, perf_hi)

    ax.plot(ai_mem, perf_mem, color="steelblue", lw=2, zorder=2)
    ax.plot(ai_cmp, perf_cmp, color="firebrick", lw=2, zorder=2)

    # Scatter: one point per config — added before canvas.draw()
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*"]
    configs = [
        (int(bs), int(td))
        for bs in sorted(df["block_size"].unique())
        for td in sorted(df["tdof"].unique())
    ]
    for ci, (bs, td) in enumerate(configs):
        row = df[(df["block_size"] == bs) & (df["tdof"] == td)]
        if row.empty:
            continue
        ai_val = float(row["arith_int"].iloc[0])
        gf_val = float(row["gflops_s"].iloc[0])
        if not (np.isfinite(ai_val) and np.isfinite(gf_val)):
            continue
        ax.scatter(
            ai_val,
            gf_val,
            color=palette(ci % 10),
            marker=markers[ci % len(markers)],
            s=60,
            zorder=4,
            label=f"B={bs}, T={td}",
        )

    # Force layout so transData gives the true display coordinates for labels
    fig.canvas.draw()

    # Memory-bound label: centered on the slope, rotated to match display angle
    p1 = ax.transData.transform((ai_mem[0], perf_mem[0]))
    p2 = ax.transData.transform((ai_mem[-1], perf_mem[-1]))
    slope_deg = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    ai_lbl_mem = 10 ** ((np.log10(ai_mem[0]) + np.log10(ai_mem[-1])) / 2)
    ax.text(
        ai_lbl_mem,
        peak_dram_gbs * ai_lbl_mem,
        f"DRAM: {peak_dram_gbs:.0f} GB/s",
        ha="center",
        va="bottom",
        fontsize=8,
        color="steelblue",
        rotation=slope_deg,
        rotation_mode="anchor",
    )

    # Compute-bound label: centered on the horizontal segment
    ai_lbl_cmp = 10 ** ((np.log10(ai_cmp[0]) + np.log10(ai_cmp[-1])) / 2)
    ax.text(
        ai_lbl_cmp,
        peak_fp32_gflops,
        f"Peak FP32: {peak_fp32_gflops / 1e3:.2f} TFLOP/s",
        ha="center",
        va="bottom",
        fontsize=8,
        color="firebrick",
    )

    ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.set_title(title)
    ax.legend(title="Kernel", fontsize=8, title_fontsize=8)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)

    fig.savefig(plot_dir / "roofline.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_throughput_bars(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    One bar-chart figure per throughput metric showing all 9 kernel configs.
    Also produces memory overview and memory-vs-compute figures.
    """
    configs = [(bs, td) for bs in BLOCK_SIZES for td in TDOFS]
    x_labels = [config_label(bs, td) for bs, td in configs]
    x = np.arange(len(configs))

    palette = plt.get_cmap("tab10")

    # -- individual metric figures: single color for all bars --
    for key, label in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
        vals = _get_vals(df, configs, key)
        bars = ax.bar(
            x, vals, color=palette(0), edgecolor="black", linewidth=0.6, width=0.65
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("% of Peak")
        ax.set_ylim(0, 110)
        ax.set_xlabel("Kernel Configuration")
        ax.set_title(label.replace("\n", " "))
        fig.savefig(plot_dir / f"throughput_{key}.pdf", bbox_inches="tight")
        plt.close(fig)

    # -- memory overview: L1, L2, Memory (overall), DRAM -- grouped bars per config --
    memory_metrics = [
        ("l1_pct", "L1"),
        ("l2_pct", "L2"),
        ("memory_pct", "Memory (Overall)"),
        ("dram_pct", "DRAM"),
    ]
    n_mem = len(memory_metrics)
    group_width = 0.8
    bar_width = group_width / n_mem
    offsets = (np.arange(n_mem) - (n_mem - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(12.0, 5.0), layout="constrained")
    for mi, (mkey, mlabel) in enumerate(memory_metrics):
        vals = _get_vals(df, configs, mkey)
        ax.bar(
            x + offsets[mi],
            vals,
            width=bar_width,
            color=palette(mi),
            edgecolor="black",
            linewidth=0.4,
            label=mlabel,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("% of Peak")
    ax.set_ylim(0, 115)
    ax.set_xlabel("Kernel Configuration")
    ax.set_title("Memory Throughput")
    ax.legend(title="Cache Level", fontsize=9)
    fig.savefig(plot_dir / "throughput_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    # -- memory (overall) vs compute -- grouped bars per config --
    fig, ax = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    bw = 0.35
    mem_vals = _get_vals(df, configs, "memory_pct")
    cmp_vals = _get_vals(df, configs, "compute_pct")
    b1 = ax.bar(
        x - bw / 2,
        mem_vals,
        width=bw,
        label="Memory (Overall)",
        color=palette(0),
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + bw / 2,
        cmp_vals,
        width=bw,
        label="Compute (SM)",
        color=palette(1),
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("% of Peak")
    ax.set_ylim(0, 115)
    ax.set_xlabel("Kernel Configuration")
    ax.set_title("Memory vs Compute Throughput")
    ax.legend()
    fig.savefig(plot_dir / "throughput_memory_vs_compute.pdf", bbox_inches="tight")
    plt.close(fig)

    # -- occupancy comparison: achieved vs theoretical --
    fig, ax = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    bw = 0.35
    achieved = [
        (
            df.loc[
                (df["block_size"] == bs) & (df["tdof"] == td), "achieved_occ"
            ].values[0]
            if len(df.loc[(df["block_size"] == bs) & (df["tdof"] == td)]) > 0
            else float("nan")
        )
        for bs, td in configs
    ]
    theoretical = [
        (
            df.loc[
                (df["block_size"] == bs) & (df["tdof"] == td), "theoretical_occ"
            ].values[0]
            if len(df.loc[(df["block_size"] == bs) & (df["tdof"] == td)]) > 0
            else float("nan")
        )
        for bs, td in configs
    ]
    b1 = ax.bar(
        x - bw / 2,
        theoretical,
        width=bw,
        label="Theoretical",
        color=palette(0),
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax.bar(
        x + bw / 2,
        achieved,
        width=bw,
        label="Achieved",
        color=palette(1),
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Occupancy [% of peak]")
    ax.set_ylim(0, 115)
    ax.set_xlabel("Kernel Configuration")
    ax.set_title(f"Theoretical vs Achieved Occupancy")
    ax.legend()
    fig.savefig(plot_dir / "occupancy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile ddh_action_kernel with ncu and plot speed-of-light statistics."
    )
    parser.add_argument(
        "--exe",
        default=str(Path("build") / "benchmarks" / "benchmark2d"),
        help="Path to benchmark2d executable.",
    )
    parser.add_argument(
        "--ncu",
        default="ncu",
        help="Path to the ncu (Nsight Compute CLI) executable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("benchmarks") / "results" / "throughput2d"),
        help="Directory where ncu logs, CSV, and plots are written.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip ncu runs and regenerate plots from existing CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exe = Path(args.exe)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    combined_csv = output_dir / "throughput2d_combined.csv"

    if args.skip_runs:
        if not combined_csv.exists():
            raise FileNotFoundError(f"Combined CSV not found: {combined_csv}")
        df = pd.read_csv(combined_csv)
        print(f"Loaded existing data from {combined_csv}")
    else:
        if not exe.exists():
            raise FileNotFoundError(f"benchmark2d executable not found: {exe}")

        configs = [(bs, td) for bs in BLOCK_SIZES for td in TDOFS]
        print(f"Profiling {len(configs)} kernel configurations with ncu ...")
        print(f"  nx=ny={NX}, p={DEGREE}, precision={PRECISION}, omega={OMEGA}")
        print(f"  warmup={WARMUP}, iterations={ITERATIONS}")

        rows = []
        failures = []
        for bs, td in configs:
            label = f"B{bs}_T{td}"
            try:
                row = run_ncu(
                    ncu=args.ncu,
                    exe=exe,
                    block_size=bs,
                    tdof=td,
                    output_dir=output_dir,
                )
                rows.append(row)
                print(
                    f"  [{label}] OK — compute={row.get('compute_pct', float('nan')):.1f}%  "
                    f"memory={row.get('memory_pct', float('nan')):.1f}%  "
                    f"dram={row.get('dram_pct', float('nan')):.1f}%  "
                    f"occ(ach/th)={row.get('achieved_occ', float('nan')):.1f}/"
                    f"{row.get('theoretical_occ', float('nan')):.1f}%"
                )
            except Exception as exc:
                failures.append({"config": label, "error": str(exc)})
                print(f"  [{label}] FAILED: {exc}", file=sys.stderr)

        if failures:
            pd.DataFrame(failures).to_csv(
                output_dir / "throughput2d_failures.csv", index=False
            )
            print(
                f"  {len(failures)} configuration(s) failed — see throughput2d_failures.csv"
            )

        if not rows:
            print("No successful runs. Exiting.", file=sys.stderr)
            sys.exit(1)

        df = pd.DataFrame(rows)
        df.to_csv(combined_csv, index=False)
        print(f"Saved combined data to {combined_csv}")

    configure_plot_style()
    plot_throughput_bars(df, plot_dir)
    plot_roofline(df, plot_dir)
    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
