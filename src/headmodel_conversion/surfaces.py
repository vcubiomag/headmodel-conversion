"""Assemble per-tissue, single-lump solids from the decimated interface complex.

The decimator returns a set of triangles (shared global vertices) each tagged
with the material pair it separates (`ref = lo*1000 + hi`, outside = 0). A
tissue's boundary is every triangle whose pair includes that tissue. Split that
boundary into connected shells; a peripheral shell plus the shells nested inside
it (its cavities -- e.g. the skull's marrow voids) is one solid. Genuinely
separate pieces (the two eyeballs) become separate solids.
"""

from __future__ import annotations

import numpy as np
import pyvista

# gmsh physical tag -> tissue label. Matches the CHARM head model.
VOLUME_KEY_TO_LABEL = {
    1: "white_matter",
    2: "gray_matter",
    3: "csf",
    5: "scalp",
    6: "eye_balls",
    7: "cortical_bone",
    8: "cancellous_bone",
    9: "blood",
    10: "muscle",
}

# Drop connected shells smaller than this many faces (decimation specks).
MIN_SHELL_FACES = 12


def _polydata(points: np.ndarray, faces: np.ndarray) -> pyvista.PolyData:
    conn = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    return pyvista.PolyData(np.ascontiguousarray(points, dtype=np.float64), conn)


def tissue_surface(
    points: np.ndarray, faces: np.ndarray, refs: np.ndarray, tag: int
) -> pyvista.PolyData:
    """The full boundary surface of one tissue, cleaned of unused vertices."""
    lo, hi = refs // 1000, refs % 1000
    mask = (lo == tag) | (hi == tag)
    surface = _polydata(points, faces[mask]).clean().triangulate()
    if surface.n_open_edges > 0:
        surface = surface.fill_holes(hole_size=1e6).clean().triangulate()
    return surface


def split_shells(surface: pyvista.PolyData) -> list[pyvista.PolyData]:
    """Split a surface into its connected closed shells, largest volume first."""
    labeled = surface.connectivity("all")
    region_ids = np.asarray(labeled.cell_data["RegionId"])
    shells = [
        labeled.extract_cells(region_ids == r).extract_surface(algorithm="dataset_surface")
        for r in np.unique(region_ids)
    ]
    shells = [s for s in shells if s.n_cells >= MIN_SHELL_FACES]
    shells.sort(key=lambda s: s.volume, reverse=True)
    return shells


def group_shells(shells: list[pyvista.PolyData]) -> list[list[pyvista.PolyData]]:
    """Group shells into solids. Each group is [outer, void, void, ...]: an
    outermost shell followed by the shells nested inside it (its cavities).
    Non-nested shells start their own group.
    """
    remaining = list(shells)  # largest first
    groups: list[list[pyvista.PolyData]] = []

    while remaining:
        outer = remaining.pop(0)
        voids: list[pyvista.PolyData] = []
        still: list[pyvista.PolyData] = []

        if remaining:
            offsets = np.cumsum([0] + [s.n_points for s in remaining])
            cloud = pyvista.PolyData(np.vstack([s.points for s in remaining]))
            inside = np.asarray(
                cloud.select_interior_points(outer, check_surface=False)["selected_points"]
            )
            for i, shell in enumerate(remaining):
                fraction = inside[offsets[i] : offsets[i + 1]].mean()
                (voids if fraction > 0.5 else still).append(shell)

        groups.append([outer, *voids])
        remaining = still

    return groups


def solid_names(label: str, groups: list[list[pyvista.PolyData]]) -> list[str]:
    """Name per solid: single -> `<label>`, multiple -> `<label>_N`.

    Eyeballs get anatomical L/R suffixes by centroid x-coordinate (RAS: +x is
    the subject's right). `groups[i][0]` is solid i's outer (peripheral) shell.
    """
    if len(groups) == 1:
        return [label]

    if label == "eye_balls" and len(groups) == 2:
        centroids_x = [float(np.asarray(g[0].points)[:, 0].mean()) for g in groups]
        order = "LR" if centroids_x[0] < centroids_x[1] else "RL"
        return [f"eye_ball_{side}" for side in order]

    return [f"{label}_{i + 1}" for i in range(len(groups))]


def tissue_solids(
    points: np.ndarray, faces: np.ndarray, refs: np.ndarray, tag: int, label: str
) -> list[tuple[str, list[pyvista.PolyData]]]:
    """All single-lump solids for one tissue, as (name, [outer, *voids])."""
    surface = tissue_surface(points, faces, refs, tag)
    if surface.n_cells == 0:
        return []
    groups = group_shells(split_shells(surface))
    if not groups:
        return []
    groups.sort(key=lambda g: g[0].n_cells, reverse=True)
    return list(zip(solid_names(label, groups), groups))


def shell_arrays(shell: pyvista.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """(points (N,3) float64, triangles (M,3) int32) for one closed shell."""
    tri = shell.triangulate().clean()
    points = np.ascontiguousarray(tri.points, dtype=np.float64)
    faces = np.asarray(tri.faces).reshape(-1, 4)
    triangles = np.ascontiguousarray(faces[:, 1:], dtype=np.int32)
    return points, triangles
