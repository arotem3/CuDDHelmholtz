import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

parser = argparse.ArgumentParser(
    description="Visualize 3D solution on boundary faces with PyVista"
)
parser.add_argument("coo", help="Path to collocation points binary file")
parser.add_argument("sol", help="Path to solution binary file (real or complex)")
parser.add_argument(
    "-o",
    "--output",
    default="solution.png",
    help="Output boundary-surface image file name (default: solution.png)",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Show interactive window (default: off-screen screenshot only)",
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
parser.add_argument(
    "--elev",
    type=float,
    default=20.0,
    help="Camera elevation angle in degrees (default: 20)",
)
parser.add_argument(
    "--azim",
    type=float,
    default=-60.0,
    help="Camera azimuth angle in degrees (default: -60)",
)
parser.add_argument(
    "--window-size",
    type=int,
    nargs=2,
    metavar=("W", "H"),
    default=(1500, 900),
    help="Render window size in pixels (default: 1500 900)",
)
parser.add_argument(
    "--lighting",
    action="store_true",
    help="Enable scene lighting (default: off for flat, clearer scalar colors)",
)
args = parser.parse_args()

sol = np.fromfile(args.sol, dtype=np.float64)
coo_raw = np.fromfile(args.coo, dtype=np.float64)

# Infer ndof and real/complex format from file sizes.
if sol.size % 2 == 0 and coo_raw.size in (3 * (sol.size // 2), 4 * (sol.size // 2)):
    ndof = sol.size // 2
    is_complex = True
else:
    ndof = sol.size
    is_complex = False

# coo.0000 may contain packed double3 (3 doubles) or padded records (4 doubles).
if coo_raw.size == 3 * ndof:
    coo = coo_raw.reshape(ndof, 3)
elif coo_raw.size == 4 * ndof:
    coo = coo_raw.reshape(ndof, 4)[:, :3]
else:
    raise ValueError(
        f"Coordinate/solution size mismatch: got {coo_raw.size} coordinate values and {sol.size} solution values. "
        f"Expected coordinate size 3*ndof={3*ndof} or 4*ndof={4*ndof}."
    )

# Structured-grid reconstruction (examples generate axis-aligned tensor-product grids).
coo_q = np.round(coo, 12)
order = np.lexsort((coo_q[:, 2], coo_q[:, 1], coo_q[:, 0]))
coo_q_sorted = coo_q[order]

x1d = np.unique(coo_q_sorted[:, 0])
y1d = np.unique(coo_q_sorted[:, 1])
z1d = np.unique(coo_q_sorted[:, 2])
nx, ny, nz = x1d.size, y1d.size, z1d.size

if nx * ny * nz != ndof:
    raise ValueError(
        "Coordinates do not form a full structured grid. "
        f"Found nx*ny*nz={nx*ny*nz} but ndof={ndof}."
    )

X, Y, Z = np.meshgrid(x1d, y1d, z1d, indexing="ij")
if not (
    np.allclose(coo_q_sorted[:, 0], X.ravel(order="C"))
    and np.allclose(coo_q_sorted[:, 1], Y.ravel(order="C"))
    and np.allclose(coo_q_sorted[:, 2], Z.ravel(order="C"))
):
    raise ValueError(
        "Sorted coordinates are not consistent with a structured axis-aligned grid."
    )

# Build fields on (nx, ny, nz) then convert to VTK point ordering (x-fastest => Fortran flatten).
if is_complex:
    uv = sol.reshape((-1, 2), order="F")
    u_re = uv[:, 0][order].reshape(nx, ny, nz)
    u_im = uv[:, 1][order].reshape(nx, ny, nz)
    u_abs = np.sqrt(u_re * u_re + u_im * u_im)
else:
    u_re = sol[order].reshape(nx, ny, nz)
    u_im = None
    u_abs = None

points = np.column_stack((X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")))

grid = pv.StructuredGrid()
grid.dimensions = (nx, ny, nz)
grid.points = points

grid.point_data["u"] = u_re.ravel(order="F")
if is_complex:
    grid.point_data["u_re"] = u_re.ravel(order="F")
    grid.point_data["u_im"] = u_im.ravel(order="F")
    grid.point_data["u_abs"] = u_abs.ravel(order="F")

# Keep only box boundary faces.
surface = grid.extract_surface(
    pass_pointid=False,
    pass_cellid=False,
    algorithm="dataset_surface",
)

xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
center = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5])
extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
up = (0.0, 0.0, 1.0)


def set_camera(plotter):
    az = math.radians(args.azim)
    el = math.radians(args.elev)
    direction = np.array(
        [
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ]
    )
    dist = 2.3 * np.linalg.norm(extent)
    position = center + dist * direction
    plotter.camera_position = [tuple(position), tuple(center), up]


plotter = pv.Plotter(
    shape=(1, 2) if is_complex else (1, 1),
    off_screen=not args.show,
    window_size=tuple(args.window_size),
)

scalar_bar_args = {
    "vertical": True,
    "width": 0.08,
    "height": 0.78,
    "position_x": 0.88,
    "position_y": 0.11,
    "title_font_size": 14,
    "label_font_size": 12,
    "n_labels": 7,
}

if is_complex:
    m = max(np.abs(surface["u_re"]).max(), np.abs(surface["u_im"]).max())
    clim = (-m, m)

    # Use distinct mesh instances so each subplot keeps its own scalar field.
    surface_re = surface.copy(deep=True)
    surface_im = surface.copy(deep=True)

    plotter.subplot(0, 0)
    plotter.add_text("Re{u}", font_size=12)
    plotter.add_mesh(
        surface_re,
        scalars="u_re",
        cmap="seismic",
        clim=clim,
        show_edges=False,
        smooth_shading=False,
        lighting=args.lighting,
        show_scalar_bar=False,
    )
    plotter.show_bounds(grid=None, all_edges=True, location="outer")
    set_camera(plotter)

    plotter.subplot(0, 1)
    plotter.add_text("Im{u}", font_size=12)
    plotter.add_mesh(
        surface_im,
        scalars="u_im",
        cmap="seismic",
        clim=clim,
        show_edges=False,
        smooth_shading=False,
        lighting=args.lighting,
        scalar_bar_args=scalar_bar_args,
    )
    plotter.show_bounds(grid=None, all_edges=True, location="outer")
    set_camera(plotter)
    plotter.link_views()
else:
    m = np.abs(surface["u"]).max()
    clim = (-m, m)

    plotter.add_text("u", font_size=12)
    plotter.add_mesh(
        surface,
        scalars="u",
        cmap="seismic",
        clim=clim,
        show_edges=False,
        smooth_shading=False,
        lighting=args.lighting,
        scalar_bar_args=scalar_bar_args,
    )
    plotter.show_bounds(grid=None, all_edges=True, location="outer")
    set_camera(plotter)

plotter.show(screenshot=args.output)
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
    pv.close_all()
