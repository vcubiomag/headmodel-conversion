"""Assemble per-tissue, single-lump solids from the interface complex.

`boundary_facets` returns a set of triangles (shared global vertices) each tagged
with the material pair it separates (`ref = lo*1000 + hi`, outside = 0) and wound
so its normal points out of `lo`. A tissue's boundary is every triangle whose
pair includes that tissue, reversed wherever the tissue is `hi` -- which leaves a
soup wound consistently out of the tissue.

`repair_solids` then splits that soup where it is non-manifold and decomposes it
into volumes: a peripheral shell plus the shells nested inside it (its cavities
-- e.g. the skull's marrow voids) is one solid, and genuinely separate pieces
(the two eyeballs) become separate solids. Ansys Maxwell disallows overlapping
dielectrics, so that cavity nesting is what the whole export rests on; it is
decided by exact predicates rather than by sampling points.

Nothing here deletes a face. A soup that arrives non-manifold -- two walls of a
tissue driven into contact -- is reported by `repair_solids` rather than patched
over: geometry that thin is a segmentation or remeshing problem to look at, not a
face to quietly drop.
"""

from __future__ import annotations

import numpy as np

from .mesher import heal_soup, repair_solids
from .sheets import split_sheets

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

# Drop shells smaller than this many faces (remeshing specks).
MIN_SHELL_FACES = 12

Shell = tuple[np.ndarray, np.ndarray]


def tissue_soup(points: np.ndarray, faces: np.ndarray, refs: np.ndarray, tag: int) -> Shell:
    """One tissue's whole boundary, wound outward, over its own vertices."""
    lo, hi = refs // 1000, refs % 1000
    mine = (lo == tag) | (hi == tag)
    tris = faces[mine]
    # Facets carry the winding out of `lo`; reverse the ones this tissue bounds
    # from the `hi` side so every normal ends up pointing out of *this* tissue.
    flip = hi[mine] == tag
    tris[flip] = tris[flip][:, [0, 2, 1]]

    used = np.unique(tris)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (
        np.ascontiguousarray(points[used], dtype=np.float64),
        np.ascontiguousarray(remap[tris], dtype=np.int32),
    )


def solid_names(label: str, groups: list[list[Shell]]) -> list[str]:
    """Name per solid: single -> `<label>`, multiple -> `<label>_N`.

    These name the products inside the tissue's STEP file, which is what tells
    its parts apart once they share one file. Eyeballs get anatomical L/R
    suffixes by centroid x-coordinate (RAS: +x is the subject's right).
    `groups[i][0]` is solid i's outer (peripheral) shell.
    """
    if len(groups) == 1:
        return [label]

    if label == "eye_balls" and len(groups) == 2:
        centroids_x = [float(g[0][0][:, 0].mean()) for g in groups]
        order = "LR" if centroids_x[0] < centroids_x[1] else "RL"
        return [f"eye_ball_{side}" for side in order]

    return [f"{label}_{i + 1}" for i in range(len(groups))]


def _repair(pts: np.ndarray, tris: np.ndarray) -> list[list[Shell]]:
    """Decompose the soup into solids, healing it first only if it will not decompose.

    Both supported methods should reach `repair_solids` with a closed manifold soup:
    CHARM's own triangulation is one already, and MMG3D remeshes a tissue's two walls
    together so it cannot drive them through each other. `heal_soup` fills tears and
    removes self-intersections, and is kept as insurance against input that violates
    that expectation -- it is expensive, so it runs only once the direct path has
    failed. A tissue reaching it is worth investigating, not ignoring.
    """
    try:
        return repair_solids(pts, tris)
    except RuntimeError:
        hp, hf, _ = heal_soup(pts, tris)
        return repair_solids(np.ascontiguousarray(hp), np.ascontiguousarray(hf))


def tissue_solids(
    points: np.ndarray, faces: np.ndarray, refs: np.ndarray, tag: int, label: str
) -> list[tuple[str, list[Shell]]]:
    """All single-lump solids for one tissue, as (name, [outer, *cavities])."""
    pts, tris = tissue_soup(points, faces, refs, tag)
    if len(tris) == 0:
        return []

    # Resolve the tissue's self-contact before handing it to CGAL, which would
    # otherwise tear it open rather than split it. See `sheets`.
    pts, tris, _ = split_sheets(pts, tris)

    groups: list[list[Shell]] = []
    for outer, *voids in _repair(pts, tris):
        if len(outer[1]) < MIN_SHELL_FACES:
            continue  # a speck, and with it any cavity it claimed to hold
        groups.append([outer, *(v for v in voids if len(v[1]) >= MIN_SHELL_FACES)])

    if not groups:
        return []
    groups.sort(key=lambda g: len(g[0][1]), reverse=True)
    return list(zip(solid_names(label, groups), groups))
