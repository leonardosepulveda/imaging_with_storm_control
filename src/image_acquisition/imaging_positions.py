import os
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from shapely.geometry import Polygon
from shapely.geometry import Point


def read_positions_file(positions_file_path):
    positions_file_dir = os.path.dirname(positions_file_path)
    points = pd.read_csv(positions_file_path, header = None, sep=',').values
    return points


def one_dim_coords(center, dmin, dmax, step_size):
    right = np.arange(center + step_size/2, dmax, step_size)
    left  = np.arange(center - step_size/2, dmin, -step_size)[::-1]
    return np.concatenate([left, right])


def one_dim_coords_even(center, dmin, dmax, step_size):
   
    span = dmax - dmin
    n = int(round(span / step_size))   # approximate number of positions

    # ensure even
    if n % 2 == 1:
        n += 1
    if n == 0:
        return np.array([])

    # calculate difference
    d = n*step_size - span
    
    # re-adjust bounds
    dmin = dmin - d/2
    dmax = dmax + d/2
        
    coords =  np.arange(dmin, dmax, step_size)

    return coords


def create_grid_positions(cx, cy, xmin, ymin, xmax, ymax, step_size):
    xs = one_dim_coords_even(cx, xmin, xmax, step_size)
    ys = one_dim_coords_even(cy, ymin, ymax, step_size)

    # ys increasing: row 0 = bottom, row -1 = top
    Xg, Yg = np.meshgrid(xs, ys)
    grid = np.stack([Xg, Yg], axis=-1)  # shape (H, W, 2)

    return grid, xs, ys


def generate_scanning_path(grid, direction="vertical"):
    """
    grid: (H, W, 2) array of positions, as from grid_positions.
    direction: "vertical" or "horizontal".
    Returns: (N, 2) array of positions in snake order.
    """
    H, W, _ = grid.shape
    path = []

    if direction == "vertical":
        # Columns: left -> right
        for j in range(W):
            if j % 2 == 0:
                # even column: top -> bottom (top row is H-1)
                rows = range(H-1, -1, -1)
            else:
                # odd column: bottom -> top
                rows = range(0, H)
            for i in rows:
                path.append(grid[i, j])
    elif direction == "horizontal":
        # Rows: top -> bottom (top row is H-1)
        for i in range(H-1, -1, -1):
            strip = (H-1 - i)  # 0 for top row, 1 for next, etc.
            if strip % 2 == 0:
                # even strip: left -> right
                cols = range(0, W)
            else:
                # odd strip: right -> left
                cols = range(W-1, -1, -1)
            for j in cols:
                path.append(grid[i, j])
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    return np.array(path)


def load_hole_polygons(hole_dir, pattern="hole*.txt"):
    """
    Read all hole*.txt files in hole_dir and return a list of Shapely Polygons.
    Each file is assumed to be a CSV with lines "X,Y".
    """
    hole_dir = Path(hole_dir)
    polygons = []

    for path in sorted(hole_dir.glob(pattern)):
        with path.open() as f:
            reader = csv.reader(f)
            coords = [(float(x), float(y)) for x, y in reader if x and y]

        if len(coords) < 3:
            print(f"Warning: {path} has fewer than 3 points; skipping.")
            continue

        poly = Polygon(coords)
        if not poly.is_empty and poly.is_valid:
            polygons.append(poly)

    return polygons



def filter_scanning_path(coords, boundary_poly, hole_polygons, dilate=0.0):
    """
    Filter snake_path coordinates so that:
      1) Points lie inside (or on) the boundary polygon (optionally dilated).
      2) Points are NOT inside any of the hole polygons.

    Parameters
    ----------
    spc : array-like of shape (N, 2)
        Snake path coordinates: [[x0, y0], [x1, y1], ...].
    boundary_poly : shapely.geometry.Polygon
        Outer boundary polygon.
    hole_polygons : list of shapely.geometry.Polygon
        List of hole polygons to exclude.
    dilate : float, optional (default=0.0)
        Buffer distance applied to the boundary polygon before checking
        points. Positive dilates (expands), negative erodes (shrinks).

    Returns
    -------
    np.ndarray of shape (M, 2)
        Filtered coordinates in the same order as input.
    """
    coords = np.asarray(coords, dtype=float)

    # Optionally dilate (buffer) the boundary
    boundary = boundary_poly.buffer(dilate) if dilate != 0.0 else boundary_poly

    kept = []
    for x, y in coords:
        p = Point(x, y)

        # 1) Must be inside or on the (possibly dilated) boundary
        if not boundary.covers(p):
            continue

        # 2) Must NOT be inside any hole
        in_hole = any(hole.covers(p) for hole in hole_polygons)
        if in_hole:
            continue

        kept.append((x, y))

    return np.array(kept)

def compute_grid_indices(spc, step_size):
    """
    Map continuous coordinates to integer grid indices (ix, iy), starting at 0.
    spc: (N, 2) array
    """
    spc = np.asarray(spc, dtype=float)
    x = spc[:, 0]
    y = spc[:, 1]

    x0 = x.min()
    y0 = y.min()

    # Convert to indices; small eps to avoid floating errors
    eps = 1e-9
    ix = ((x - x0) / step_size + eps).astype(int)
    iy = ((y - y0) / step_size + eps).astype(int)

    return ix, iy


def select_side_indices(spc, ix, iy, return_side):
    """
    Given snake-path coordinates and their grid indices, return the indices
    of points on the chosen side: 'top', 'bottom', 'left', 'right'.
    """
    spc = np.asarray(spc)
    return_side = return_side.lower()
    if return_side not in {"top", "bottom", "left", "right"}:
        raise ValueError("return_side must be one of: 'top', 'bottom', 'left', 'right'")

    indices = np.arange(spc.shape[0])

    selected = []

    if return_side in {"top", "bottom"}:
        # Group by column (ix)
        for col in np.unique(ix):
            mask = ix == col
            col_indices = indices[mask]
            col_iy = iy[mask]

            if return_side == "top":
                # max iy in this column
                k = np.argmax(col_iy)
            else:  # bottom
                k = np.argmin(col_iy)

            selected.append(col_indices[k])

    else:  # 'left' or 'right'
        # Group by row (iy)
        for row in np.unique(iy):
            mask = iy == row
            row_indices = indices[mask]
            row_ix = ix[mask]

            if return_side == "right":
                # max ix in this row
                k = np.argmax(row_ix)
            else:  # left
                k = np.argmin(row_ix)

            selected.append(row_indices[k])

    # Deduplicate and sort in path order
    selected = sorted(set(selected))
    return np.array(selected, dtype=int)


def close_scanning_path(spc, step_size, return_side):
    """
    Reorder snake-path coordinates so that the points on the requested side
    ('top', 'bottom', 'left', 'right') are moved to the end of the sequence,
    in reversed relative order.

    Additionally:
      - If index 0 (the first point) is part of that side set, it is *not* moved.

    Returns:
        new_spc: (N, 2) array with reordered coordinates
        side_indices: original indices that were moved (after removing 0 if present)
    """
    spc = np.asarray(spc, dtype=float)
    N = spc.shape[0]

    ix, iy = compute_grid_indices(spc, step_size)
    side_idxs = select_side_indices(spc, ix, iy, return_side)

    # Remove index 0 from the side set, if present
    side_idxs = side_idxs[side_idxs != 0]

    # All indices
    all_idxs = np.arange(N)

    # Indices that stay in place (relative order preserved)
    stay_mask = np.ones(N, dtype=bool)
    stay_mask[side_idxs] = False
    stay_idxs = all_idxs[stay_mask]

    # Side indices appended in reversed relative order
    moved_idxs = side_idxs[::-1]

    new_order = np.concatenate([stay_idxs, moved_idxs])
    new_spc = spc[new_order]

    return new_spc, side_idxs


def get_path_stats(snake_path_coordinates):
    """
    Compute total path length and largest distance between
    consecutive points in a path.

    Parameters
    ----------
    snake_path_coordinates : array-like of shape (N, 2)
        Ordered coordinates of the path.

    Returns
    -------
    total_length : float
        Sum of Euclidean distances between consecutive points.
    max_step : float
        Maximum Euclidean distance between any two consecutive points.
    """
    coords = np.asarray(snake_path_coordinates, dtype=float)

    if coords.shape[0] < 2:
        return 0.0, 0.0

    # Differences between consecutive points
    diffs = coords[1:] - coords[:-1]          # shape (N-1, 2)
    dists = np.linalg.norm(diffs, axis=1)     # shape (N-1,)

    total_length = dists.sum()
    max_step = dists.max()

    return total_length, max_step