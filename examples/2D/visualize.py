import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

parser = argparse.ArgumentParser(
    description="Visualize 2D solution (Poisson, Helmholtz, WaveHoltz, DDH)"
)
parser.add_argument("xy", help="Path to xy binary file")
parser.add_argument("sol", help="Path to solution binary file (real or complex)")
parser.add_argument(
    "--grid",
    type=int,
    default=200,
    metavar="N",
    help="Grid resolution N for NxN interpolation grid (default: 200)",
)
parser.add_argument(
    "-o",
    "--output",
    default="solution.png",
    help="Output file name (default: solution.png)",
)
parser.add_argument(
    "--show", action="store_true", help="Display the figure interactively after saving"
)
parser.add_argument(
    "--residuals",
    help="Optional path to FP64 binary residuals file to plot residual vs iteration",
)
parser.add_argument(
    "--residual-output",
    default="residuals.png",
    help="Output file name for residual plot (default: residuals.png)",
)
args = parser.parse_args()

xy = np.fromfile(args.xy, dtype=np.float64)
x, y = xy.reshape((2, -1), order="F")
ndof = x.size

sol = np.fromfile(args.sol, dtype=np.float64)
is_complex = sol.size == 2 * ndof

if is_complex:
    u, v = sol.reshape((-1, 2), order="F").T
else:
    u = sol
    v = None

xi = np.linspace(x.min(), x.max(), args.grid)
yi = np.linspace(y.min(), y.max(), args.grid)
Xi, Yi = np.meshgrid(xi, yi)
extent = [x.min(), x.max(), y.min(), y.max()]

points = np.column_stack((x, y))
interp_u = LinearNDInterpolator(points, u)
Ui = interp_u(Xi, Yi)

if is_complex:
    interp_v = LinearNDInterpolator(points, v)
    Vi = interp_v(Xi, Yi)

    mu = np.nanmax(np.abs(Ui))
    mv = np.nanmax(np.abs(Vi))
    m = max(mu, mv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300, layout="constrained")

    im1 = ax1.imshow(
        Ui,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=-m,
        vmax=m,
        cmap="seismic",
        interpolation="bilinear",
    )
    ax1.set_title(r"$\Re\{u\}$")

    im2 = ax2.imshow(
        Vi,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=-m,
        vmax=m,
        cmap="seismic",
        interpolation="bilinear",
    )
    fig.colorbar(im2, ax=ax2, shrink=0.6)
    ax2.set_title(r"$\Im\{u\}$")
    ax2.set_yticks([])
else:
    m = np.nanmax(np.abs(Ui))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300, layout="constrained")

    im = ax.imshow(
        Ui,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=-m,
        vmax=m,
        cmap="seismic",
        interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(r"$u$")

plt.savefig(args.output)
print(f"Saved figure to {args.output}")

if args.residuals:
    residuals = np.fromfile(args.residuals, dtype=np.float64)
    if residuals.size > 0:
        fig_res, ax_res = plt.subplots(figsize=(6, 4), dpi=300, layout="constrained")
        iterations = np.arange(1, residuals.size + 1)
        ax_res.semilogy(iterations, residuals)
        ax_res.set_xlabel("Iteration")
        ax_res.set_ylabel("Relative Residual")
        ax_res.grid(True, which="both", alpha=0.35)
        fig_res.savefig(args.residual_output)
        print(f"Saved residual plot to {args.residual_output}")

if args.show:
    plt.show()
